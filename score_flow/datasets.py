# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
from typing import Any, Optional, Tuple
import os

import jax
import ml_collections
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import tensorflow_datasets as tfds

SUPPORTED_DATASETS = ['CIFAR10', 'CELEBA', 'fastMRI', 'SVHN', 'LSUN', 'CelebAHQ', 'SgrA', 'GRMHD', 'Pynoisy',
                      'Matern', 'Dispersion', 'Burgers']


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset_builder_and_resize_op(config: ml_collections.ConfigDict, shuffle_seed = None) -> Tuple[Any, Any]:
  """Create dataset builder and image resizing function for dataset."""
  data_dir = config.data.tfds_dir
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10', data_dir=data_dir)

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      if config.data.num_channels == 1:
        img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size],
          antialias=config.data.antialias)

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a', data_dir=data_dir)

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      if config.data.num_channels == 1:
        img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size],
          antialias=config.data.antialias)

  elif config.data.dataset == 'fastMRI':
    features_dict = {
      'image': tf.io.FixedLenFeature([320*320], tf.float32),
      'shape': tf.io.FixedLenFeature([2], tf.int64)
    }
    def parse_example(serialized_example):
      parsed_example = tf.io.parse_single_example(serialized_example, features=features_dict)
      shape = parsed_example['shape']
      parsed_example['image'] = tf.reshape(parsed_example['image'], (shape[0], shape[1], 1))
      return parsed_example
    
    def ds_from_tfrecords(tfrecords_pattern):
      shard_files = tf.io.matching_files(tfrecords_pattern)
      # shard_files = tf.random.shuffle(shard_files)
      shard_files = tf.random.shuffle(shard_files, seed=shuffle_seed)
      shards = tf.data.Dataset.from_tensor_slices(shard_files)
      ds = shards.interleave(tf.data.TFRecordDataset)
      ds = ds.map(
        map_func=parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return ds

    dataset_builder = {
      'train': ds_from_tfrecords(os.path.join(config.data.tfds_dir, 'fastmri/fastmri-train.tfrecord-*')),
      'val': ds_from_tfrecords(os.path.join(config.data.tfds_dir, 'fastmri/fastmri-val.tfrecord-*')),
      'test': ds_from_tfrecords(os.path.join(config.data.tfds_dir, 'fastmri/fastmri-test.tfrecord-*')),
    }

    def resize_op(img):
      if config.data.num_channels == 3:
        img = tf.image.grayscale_to_rgb(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size],
          antialias=config.data.antialias)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped', data_dir=data_dir)

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      if config.data.num_channels == 1:
        img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(
        img, [config.data.image_size, config.data.image_size],
        antialias=config.data.antialias)

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}', data_dir=data_dir)

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        if config.data.num_channels == 1:
          img = tf.image.rgb_to_grayscale(img)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        if config.data.num_channels == 1:
          img = tf.image.rgb_to_grayscale(img)
        return img

  elif config.data.dataset == 'CelebAHQ':
    def ds_from_tfrecords(tfrecords_pattern):
      shard_files = tf.io.matching_files(tfrecords_pattern)
      shard_files = tf.random.shuffle(shard_files, seed=shuffle_seed)
      shards = tf.data.Dataset.from_tensor_slices(shard_files)
      ds = shards.interleave(tf.data.TFRecordDataset)
      return ds
  
    dataset_builder = {
      'train': tf.data.TFRecordDataset(os.path.join(config.data.tfds_dir, 'celebahq_256/celebahq_256_train-r08.tfrecords')),
      'val': tf.data.TFRecordDataset(os.path.join(config.data.tfds_dir, 'celebahq_256/celebahq_256_test-r08.tfrecords')),
      'test': tf.data.TFRecordDataset(os.path.join(config.data.tfds_dir, 'celebahq_256/celebahq_256_test-r08.tfrecords')),
    }
    def resize_op(img):
      if config.data.num_channels == 1:
        img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size],
          antialias=config.data.antialias)
  elif config.data.dataset in ['SgrA', 'GRMHD', 'PinknoiseFull', 'Pinknoise', 'Pynoisy', 'Matern', 'Dispersion', 'Burgers']:
    if config.data.dataset == 'SgrA':
      image_dim = 100 * 100
      dataset_name = 'sgra'
    elif config.data.dataset == 'GRMHD':
      image_dim = 400 * 400
      dataset_name = 'grmhd'
    elif config.data.dataset == 'Pynoisy':
      image_dim = 160 * 160
      dataset_name = 'pynoisy'
    elif config.data.dataset == 'Matern':
      image_dim = 32 * 32
      dataset_name = f'matern{config.data.matern_scale}'
    elif config.data.dataset == 'Dispersion':
      image_dim = config.data.image_size * config.data.image_size
      dataset_name = f'dispersion{config.data.image_size}'
    elif config.data.dataset == 'Burgers':
      image_dim = config.data.image_size * config.data.image_size
      dataset_name = 'burgers'
    features_dict = {
      'image': tf.io.FixedLenFeature([image_dim], tf.float32),
      'shape': tf.io.FixedLenFeature([2], tf.int64)
    }
    def parse_example(serialized_example):
      parsed_example = tf.io.parse_single_example(serialized_example, features=features_dict)
      shape = parsed_example['shape']
      parsed_example['image'] = tf.reshape(parsed_example['image'], (shape[0], shape[1], 1))
      return parsed_example
    
    def ds_from_tfrecords(tfrecords_pattern):
      shard_files = tf.io.matching_files(tfrecords_pattern)
      shard_files = tf.random.shuffle(shard_files, seed=shuffle_seed)
      shards = tf.data.Dataset.from_tensor_slices(shard_files)
      ds = shards.interleave(tf.data.TFRecordDataset)
      ds = ds.map(
        map_func=parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return ds
    
    if config.data.dataset == 'Matern':
      dataset_builder = {
        'train': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'matern/{dataset_name}/{dataset_name}-train.tfrecord-*')),
        'val': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'matern/{dataset_name}/{dataset_name}-val.tfrecord-*')),
        'test': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'matern/{dataset_name}/{dataset_name}-test.tfrecord-*')),
      }
    elif config.data.dataset == 'Dispersion':
      dataset_builder = {
        'train': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'dispersion/{dataset_name}/{dataset_name}-train.tfrecord-*')),
        'val': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'dispersion/{dataset_name}/{dataset_name}-val.tfrecord-*')),
        'test': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'dispersion/{dataset_name}/{dataset_name}-test.tfrecord-*')),
      }
    elif config.data.dataset == 'Burgers':
      image_size = config.data.image_size
      dataset_builder = {
        'train': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'burgers/burgers{image_size}x{image_size}/{dataset_name}-train.tfrecord-*')),
        'val': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'burgers/burgers{image_size}x{image_size}/{dataset_name}-val.tfrecord-*')),
        'test': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'burgers/burgers{image_size}x{image_size}/{dataset_name}-test.tfrecord-*')),
      }
    else:
      dataset_builder = {
        'train': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'{dataset_name}/{dataset_name}-train.tfrecord-*')),
        'val': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'{dataset_name}/{dataset_name}-val.tfrecord-*')),
        'test': ds_from_tfrecords(
          os.path.join(config.data.tfds_dir,
                       f'{dataset_name}/{dataset_name}-test.tfrecord-*')),
      }

    def resize_op(img):
      if config.data.num_channels == 3:
        img = tf.image.grayscale_to_rgb(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size],
          antialias=config.data.antialias)
      
  else:
    raise ValueError(
        f'Dataset {config.data.dataset} not supported.')
  return dataset_builder, resize_op


def get_preprocess_fn(config: ml_collections.ConfigDict,
                      resize_op: Any,
                      uniform_dequantization: bool = False,
                      evaluation: bool = False) -> Any:
  """Create preprocessing function for dataset."""

  # Get function for tapering images with a centered Gaussian blob.
  taper_fn = get_taper_fn(config)
  # warp_fn = get_warp_fn(config)

  if config.data.dataset == 'CelebAHQ':
    @tf.autograph.experimental.do_not_convert
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)
  elif config.data.dataset == 'SgrA' or config.data.dataset == 'GRMHD':
    # These datasets' images can be randomly rotated and zoomed in/out.
    @tf.autograph.experimental.do_not_convert
    def preprocess_fn(d):
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if config.data.random_rotation and not evaluation:
        img = tf.keras.layers.RandomRotation(
          factor=(-1., 1.), fill_mode='constant', fill_value=0.)(img)
      if config.data.random_zoom and not evaluation:
        print('USING RANDOM ZOOM')
        # Assume that GRMHD and RIAF images have ring diameter 40 uas and
        # FOV 128 uas. We want ring diameters between 35 and 48 uas.
        img = tf.keras.layers.RandomZoom(
          height_factor=(-0.167, 0.145), fill_mode='constant', fill_value=0.)(img)
        #   # M87 diameter is 40 uas. We want ring diameters between 35 and 60 uas.
        #   # img = tf.keras.layers.RandomZoom(
        #   #   height_factor=(-0.5, 0.25), fill_mode='constant', fill_value=0.)(img)
        #   # SgrA diameter is ~50 uas. We want ring diameters between 35 and 60 uas.
        #   img = tf.keras.layers.RandomZoom(
        #     height_factor=(-0.2, 0.3), fill_mode='constant', fill_value=0.)(img)
        # else:
        #   img = tf.keras.layers.RandomZoom(
        #     height_factor=(-0.1, 0.1), fill_mode='constant', fill_value=0.)(img)
      if config.data.constant_flux:
        img *= config.data.total_flux / tf.reduce_sum(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=d.get('label', None))
  else:
    @tf.autograph.experimental.do_not_convert
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      if config.data.cifartap64:
        # For cifartap64 prior, we start with 32x32, pad to 64x64, and blur edges.
        img = tf.image.convert_image_dtype(d['image'], tf.float32)
        if config.data.num_channels == 1:
          img = tf.image.rgb_to_grayscale(img)
        img = tf.pad(img, tf.constant([[16, 16], [16, 16], [0, 0]], tf.int32), 'CONSTANT')
      else:
        img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if config.data.warp:
        img = tf.numpy_function(warp_fn, [img], tf.float32)
      if config.data.taper:
        img = tf.numpy_function(taper_fn, [img], tf.float32)
      if config.data.constant_flux:
        img *= config.data.total_flux / tf.reduce_sum(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=d.get('label', None))

  return preprocess_fn


def get_dataset(
    config: ml_collections.ConfigDict,
    additional_dim: Optional[int] = None,
    uniform_dequantization: bool = False,
    evaluation: bool = False,
    shuffle_seed: Optional[int] = None,
    device_batch: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """Create data loaders for training, validation, and testing.
  Most of the logic from `score_sde/datasets.py` is kept.
  Args:
    config: The config.
    additional_dim: If not `None`, add an additional dimension
      to the output data for jitted steps.
    uniform_dequantization: If `True`, uniformly dequantize the images.
      This is usually only used when evaluating log-likelihood [bits/dim]
      of the data.
    evaluation: If `True`, fix number of epochs to 1.
    shuffle_seed: Optional seed for shuffling dataset.
    device_batch: If `True`, divide batch size into device batch and
      local batch.
  Returns:
    train_ds, val_ds, test_ds.
  """
  if config.data.dataset not in SUPPORTED_DATASETS:
    raise NotImplementedError(
        f'Dataset {config.data.dataset} not yet supported.')

  # Compute batch size for this worker.
  batch_size = (
      config.training.batch_size if not evaluation else config.eval.batch_size)
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by '
                     f'the number of devices ({jax.device_count()})')

  per_device_batch_size = batch_size // jax.device_count()
  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1
  # Create additional data dimension when jitting multiple steps together
  if not device_batch:
    batch_dims = [batch_size]
  elif additional_dim is None:
    batch_dims = [jax.local_device_count(), per_device_batch_size]
  else:
    batch_dims = [
        jax.local_device_count(), additional_dim, per_device_batch_size
    ]

  # Get dataset builder.
  dataset_builder, resize_op = get_dataset_builder_and_resize_op(config, shuffle_seed)

  # Get preprocessing function.
  preprocess_fn = get_preprocess_fn(
      config, resize_op, uniform_dequantization, evaluation)

  def create_dataset(dataset_builder: Any,
                     split: str,
                     take_val_from_train: bool = False,
                     train_split: float = 0.9):
    # Some datasets only include train and test sets, in which case we take
    # validation data from the training set.
    if split == 'test':
      take_val_from_train = False
    source_split = 'train' if take_val_from_train else split

    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = 48
    dataset_options.threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(
        options=dataset_options, shuffle_seed=shuffle_seed)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
          split=source_split, shuffle_files=True, read_config=read_config)
    elif config.data.dataset in [
        'Eigenfaces', 'fastMRI', 'CelebAHQ', 'SgrA', 'GRMHD', 'Pynoisy', 'Matern', 'Dispersion', 'Burgers'
    ]:
      ds = dataset_builder[source_split].with_options(dataset_options)
    else:
      ds = dataset_builder.with_options(dataset_options)

    if take_val_from_train:
      train_size = int(train_split * len(ds))
      # Take the first `train_split` pct. for training and the rest for val.
      ds = ds.take(train_size) if split == 'train' else ds.skip(train_size)

    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  # Set the correct split names.
  if config.data.dataset == 'CIFAR10':
    train_ds = create_dataset(
        dataset_builder, 'train', take_val_from_train=True)  # 50,000 * 0.9
    val_ds = create_dataset(
        dataset_builder, 'validation', take_val_from_train=True)  # 50,000 * 0.1
    test_ds = create_dataset(dataset_builder, 'test')  # 10,000
  elif config.data.dataset == 'CELEBA':
    train_ds = create_dataset(dataset_builder, 'train')  # 162,770
    test_ds = create_dataset(dataset_builder, 'test')  # 19,962
    val_ds = create_dataset(dataset_builder, 'validation')  # 19,867
  elif config.data.dataset == 'fastMRI':
    train_ds = create_dataset(dataset_builder, 'train')  # 30,199
    test_ds = create_dataset(dataset_builder, 'test')  # 571
    val_ds = create_dataset(dataset_builder, 'val')  # 5,579
  elif config.data.dataset == 'SVHN':
    train_ds = create_dataset(dataset_builder, 'train')  # 73,257
    test_ds = create_dataset(dataset_builder, 'extra')  # 531,131
    val_ds = create_dataset(dataset_builder, 'test')  # 26,032
  elif config.data.dataset == 'LSUN':
    train_ds = create_dataset(
        dataset_builder, 'train', take_val_from_train=True)  # 50,000 * 0.9
    val_ds = create_dataset(
        dataset_builder, 'validation', take_val_from_train=True)  # 50,000 * 0.1
    test_ds = create_dataset(dataset_builder, 'validation')  # 10,000
  elif config.data.dataset == 'CelebAHQ':
    # NOTE: This assumes a validation set is not used during training, so
    # uses the full training set for training.
    train_ds = create_dataset(
        dataset_builder, 'train', take_val_from_train=False)  # 29,990
    val_ds = create_dataset(
        dataset_builder, 'val', take_val_from_train=False)  # N/A
    test_ds = create_dataset(dataset_builder, 'test')  # 10
  elif config.data.dataset == 'SgrA':
    train_ds = create_dataset(dataset_builder, 'train')  # 9,070
    test_ds = create_dataset(dataset_builder, 'test')  # 10
    val_ds = create_dataset(dataset_builder, 'val')  # 10
  elif config.data.dataset == 'GRMHD':
    train_ds = create_dataset(dataset_builder, 'train')  # 100,000
    test_ds = create_dataset(dataset_builder, 'test')  # 100
    val_ds = create_dataset(dataset_builder, 'val')  # 100
  elif config.data.dataset == 'Matern':
    train_ds = create_dataset(dataset_builder, 'train')  # 45,000
    test_ds = create_dataset(dataset_builder, 'test')  # 10,000
    val_ds = create_dataset(dataset_builder, 'val')  # 5,000
  elif config.data.dataset == 'Pynoisy':
    train_ds = create_dataset(dataset_builder, 'train')  # 12,000
    test_ds = create_dataset(dataset_builder, 'test')  # 100
    val_ds = create_dataset(dataset_builder, 'val')  # 100
  elif config.data.dataset == 'Dispersion':
    train_ds = create_dataset(dataset_builder, 'train')  # 14,800
    test_ds = create_dataset(dataset_builder, 'test')  # 10
    val_ds = create_dataset(dataset_builder, 'val')  # 10
  elif config.data.dataset == 'Burgers':
    train_ds = create_dataset(dataset_builder, 'train')  # 10,000
    test_ds = create_dataset(dataset_builder, 'test')  # 1,000
    val_ds = create_dataset(dataset_builder, 'val')  # 1,000
  return train_ds, val_ds, test_ds


def get_warp_fn(config):
  import cv2
  from scipy.ndimage.interpolation import map_coordinates

  image_size = config.data.image_size
  image_shape = (image_size, image_size, config.data.num_channels)
  alpha = config.data.warp_alpha
  sigma = config.data.warp_sigma
  alpha_affine = config.data.warp_alpha_affine

  def elastic_transform(image):
    center_square = np.float32((image_size, image_size)) // 2
    square_size = image_size // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(
      image, M, (image_size, image_size), borderMode=cv2.BORDER_REFLECT_101)
    warped = np.expand_dims(warped, axis=-1)
    dx = gaussian_filter((np.random.rand(*image_shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*image_shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(
      np.arange(image_shape[1]),
      np.arange(image_shape[0]),
      np.arange(image_shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    warped = map_coordinates(warped, indices, order=1, mode='reflect').reshape(image_shape)
    return warped
  
  return elastic_transform


def get_taper_fn(config):

  def make_circle_image(frac_radius_min, frac_radius_max):
    frac_radius = np.random.uniform(frac_radius_min, frac_radius_max)
    x = np.linspace(-1, 1, config.data.image_size, dtype=np.float32)
    y = np.linspace(-1, 1, config.data.image_size, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x * x + y * y)
    mask = np.zeros((config.data.image_size, config.data.image_size), dtype=np.float32)
    mask[d <= frac_radius] = 1
    # if config.data.taper_frac_blur:
    #   sigma = config.data.taper_gaussian_blur_sigma * frac_radius * config.data.image_size
    # else:
    #   sigma = config.data.taper_gaussian_blur_sigma
    # mask = gaussian_filter(mask, sigma, mode='constant', cval=0)
    return mask

  def make_square_image(frac_radius_min, frac_radius_max):
    frac_radius = np.random.uniform(frac_radius_min, frac_radius_max)
    height = width = config.data.image_size
    radius = int(frac_radius * config.data.image_size // 2)
    mask = np.zeros((config.data.image_size, config.data.image_size), dtype=np.float32)
    mask[width // 2 - radius:width // 2 + radius, :][:, height // 2 - radius:height // 2 + radius] = 1
    # if config.data.taper_frac_blur:
    #   sigma = config.data.taper_gaussian_blur_sigma * frac_radius * config.data.image_size
    # else:
    #   sigma = config.data.taper_gaussian_blur_sigma
    # mask = gaussian_filter(mask, sigma, mode='constant', cval=0)
    return mask

  def taper_fn(image):
    if config.data.circular_taper:
      mask = make_circle_image(
        config.data.taper_frac_radius_min, config.data.taper_frac_radius_max)
    else:
      mask = make_square_image(
        config.data.taper_frac_radius_min, config.data.taper_frac_radius_max)
    # Apply Gaussian blur to hard mask.
    mask = gaussian_filter(
      mask, config.data.taper_gaussian_blur_sigma, mode='constant', cval=0)
    mask = mask / mask.max()
    mask[mask < 1e-3] = 0
    # Add channel axis.
    mask = np.expand_dims(mask, axis=-1)
    tapered_image = image * mask
    return tapered_image

  return taper_fn