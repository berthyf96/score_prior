# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Utils for score_prior."""
import math
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
from PIL import Image
import scipy
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

from score_flow import sde_lib


def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


def plot_image_grid(y, nrow=None, padding=0.1, title='', cmap='gray', normalize=(0, 1), figsize=(10, 10)):
  """Plot an image grid with matplotlib.
  
  Source: https://github.com/ameroyer/glow_jax/blob/main/train.ipynb.
  """
  images = np.clip(y, 0., 1.) if y.shape[-1] == 3 else y
  fig = plt.figure(figsize=figsize)
  fig.suptitle(title, fontsize=20)
  if nrow is None:
    nrow = int(np.floor(np.sqrt(images.shape[0])))
  ncol = len(y) // nrow
  grid = ImageGrid(fig, 111, nrows_ncols=(nrow, ncol), axes_pad=padding)
  for ax in grid: 
    ax.set_axis_off()
  for ax, im in zip(grid, images):
    ax.imshow(im, cmap=cmap, norm=Normalize(*normalize))
  fig.subplots_adjust(top=0.98)
  plt.show()
  return fig

def get_sde(config
            ):
  """Return the SDE and time-0 epsilon based on the given config."""
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(
        beta_min=config.model.beta_min, beta_max=config.model.beta_max,
        N=config.model.num_scales)
    t0_eps = 1e-3  # epsilon for stability near time 0
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(
        beta_min=config.model.beta_min, beta_max=config.model.beta_max,
        N=config.model.num_scales)
    t0_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(
        sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
        N=config.model.num_scales)
    t0_eps = 1e-5
  else:
    raise NotImplementedError(f'SDE {config.training.sde} unknown.')
  t0_eps = config.training.smallest_time
  return sde, t0_eps


def get_marginal_dist_fn(config
                         ):
  """Return a function that gives the scale and std. dev. of $p_0t$.
  See https://github.com/yang-song/score_sde/blob/main/sde_lib.py.
  `alpha_t` and `beta_t` are determined by the method
  `score_sde.sde_lib.SDE.marginal_prob`, where `alpha_t` is the coefficient of
  the mean, and `beta_t` is the std. dev.
  Args:
    config: An ml_collections.ConfigDict with the SDE configuration.
  Returns:
    _marginal_dist_fn: A callable that returns the mean coefficient `alpha_t`
      and std. dev. `beta_t` for a given diffusion time `t`.
  """
  if config.training.sde.lower() == 'vpsde':
    beta_0, beta_1 = config.model.beta_min, config.model.beta_max
    def _marginal_dist_fn(t):
      log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
      alpha_t = jnp.exp(log_mean_coeff)
      beta_t = jnp.sqrt(1 - jnp.exp(2. * log_mean_coeff))
      return alpha_t, beta_t

  elif config.training.sde.lower() == 'vesde':
    sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
    def _marginal_dist_fn(t):
      alpha_t = jnp.ones_like(t)
      beta_t = sigma_min * (sigma_max / sigma_min) ** t
      return alpha_t, beta_t

  elif config.training.sde.lower() == 'subvpsde':
    beta_0, beta_1 = config.model.beta_min, config.model.beta_max
    def _marginal_dist_fn(t):
      log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
      alpha_t = jnp.exp(log_mean_coeff)
      beta_t = 1 - jnp.exp(2. * log_mean_coeff)
      return alpha_t, beta_t
  else:
    raise NotImplementedError(f'Unsupported SDE: {config.training.sde}')

  return _marginal_dist_fn


def psplit(
    rng
):
  """Split a JAX RNG into pmapped RNGs."""
  rng, *step_rngs = jax.random.split(rng, jax.local_device_count() + 1)
  step_rngs = jnp.asarray(step_rngs)
  return rng, step_rngs


def gaussian_logp(x, mu, sigma):
  """Evaluates the log-probability of x under N(mu, sigma**2)."""
  dim = x.size
  return (-dim / 2. * jnp.log(2 * jnp.pi * sigma**2) - jnp.sum((x - mu)**2) /
          (2 * sigma**2))


def save_image_grid(ndarray,
                    fp,
                    nrow = 8,
                    padding = 2,
                    image_format = None):
  """Make a grid of images and save it into an image file.
  This implementation is modified from the one in
  https://github.com/yang-song/score_sde/blob/main/utils.py.
  Pixel values are assumed to be within [0, 1].
  Args:
    ndarray: 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow: Number of images displayed in each row of the grid.
      The final grid size is ``(nrow, B // nrow)``.
    padding: Amount of zero-padding on each image.
    image_format:  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """
  if not (isinstance(ndarray, (jnp.ndarray, np.ndarray)) or
          (isinstance(ndarray, list) and
           all(isinstance(t, (jnp.ndarray, np.ndarray)) for t in ndarray))):
    raise TypeError('array_like of tensors expected, got {}'.format(
        type(ndarray)))
  ndarray = np.asarray(ndarray)

  # Keep largest-possible number of images for given `nrow`.
  ncol = len(ndarray) // nrow
  ndarray = ndarray[:nrow * ncol]

  def _pad(image):
    # Pads a 3D array in the height and width dimensions.
    return np.pad(image, ((padding, padding), (padding, padding), (0, 0)))

  grid = np.concatenate([
      np.concatenate([
          _pad(im) for im in ndarray[row * ncol:(row + 1) * ncol]], axis=1)
      for row in range(nrow)], axis=0)

  # For grayscale images, need to remove the third axis.
  grid = np.squeeze(grid)

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer.
  ndarr = np.clip(grid * 255. + 0.5, 0, 255).astype(np.uint8)

  im = Image.fromarray(ndarr)
  im.save(fp, format=image_format)


def is_coordinator():
  return jax.process_index() == 0


def convert_tb_data(root_dir, sort_by=None):
  """Convert local TensorBoard data into Pandas DataFrame.

  Function takes the root directory path and recursively parses
  all events data.    
  If the `sort_by` value is provided then it will use that column
  to sort values; typically `wall_time` or `step`.

  *Note* that the whole data is converted into a DataFrame.
  Depending on the data size this might take a while. If it takes
  too long then narrow it to some sub-directories.

  Paramters:
      root_dir: (str) path to root dir with tensorboard data.
      sort_by: (optional str) column name to sort by.

  Returns:
      pandas.DataFrame with [wall_time, name, step, value] columns.

  """
  def convert_tfevent(filepath):
    return pd.DataFrame([
        parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
    ])

  def parse_tfevent(tfevent):
    return dict(
        wall_time=tfevent.wall_time,
        name=tfevent.summary.value[0].tag,
        step=tfevent.step,
        value=tf.make_ndarray(tfevent.summary.value[0].tensor).item(),
    )

  columns_order = ['wall_time', 'name', 'step', 'value']

  out = []
  for (root, _, filenames) in os.walk(root_dir):
    for filename in filenames:
      if "events.out.tfevents" not in filename:
        continue
      file_full_path = os.path.join(root, filename)
      out.append(convert_tfevent(file_full_path))

  # Concatenate (and sort) all partial individual dataframes
  all_df = pd.concat(out)[columns_order]
  if sort_by is not None:
    all_df = all_df.sort_values(sort_by)

  return all_df.reset_index(drop=True)


def smooth(scalars, weight):
  """
  EMA implementation according to
  https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
  """
  last = 0
  smoothed = []
  num_acc = 0
  for next_val in scalars:
    last = last * weight + (1 - weight) * next_val
    num_acc += 1
    # de-bias
    debias_weight = 1
    if weight != 1:
        debias_weight = 1 - math.pow(weight, num_acc)
    smoothed_val = last / debias_weight
    smoothed.append(smoothed_val)
  return np.array(smoothed)


def is_converged(values, smoothing_kernel_width,
                 decrease_thresh=-1e-6, increase_thresh=1e-3, patience=3,
                 min_num_values=100):
  """Check convergence of the given list of loss values.
  
  Args:
    values: List or array of loss values.
    smoothing_kernel_width: Width of kernel for Gaussian filtering of loss curve.
    decrease_thresh: If last `patience` number of steps all showed a relative
      decrease less than `decrease_thresh`, then consider `values` converged.
    increase_thresh: If last `patience` number of steps all showed a relative
      increase greater than `increase_thresh`, then consider `values` converged.
    patience: Do not consider converged until the last `patience` number of
      steps all meet the convergence criterion.
    min_num_values: Only consider convergence once at least `min_num_values`
      are given.
  """
  if len(values) < min_num_values:
    return False
  smoothed = scipy.ndimage.gaussian_filter(values, smoothing_kernel_width)
  rel_diffs = (smoothed[1:] - smoothed[:-1]) / abs(smoothed[:-1])
  if np.all(rel_diffs[-patience:] > increase_thresh):
    return True
  if np.all(rel_diffs[-patience:] < 0) and np.all(abs(rel_diffs[-patience:]) < decrease_thresh):
    return True
  return False


def precondition_covariance(cov, eps=1e-3):
  eigvals, eigvecs = np.linalg.eigh(cov)
  eigvals += eps
  cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
  return cov


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    grad_fn = lambda data: jnp.sum(fn(data, t) * eps)
    grad_fn_eps = jax.grad(grad_fn)(x)
    return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

  return div_fn


def get_value_div_fn(fn):
  """Return both the function value and its estimated divergence via Hutchinson's trace estimator."""

  def value_div_fn(x, t, eps):
    def value_grad_fn(data):
      f = fn(data, t)
      return jnp.sum(f * eps), f
    grad_fn_eps, value = jax.grad(value_grad_fn, has_aux=True)(x)
    return value, jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

  return value_div_fn

def plot_tight_image(image, **kwargs):
  """Plot an image without any padding."""
  fig = plt.figure(frameon=False)
  fig.set_size_inches(image.shape[0], image.shape[1])
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(image, aspect='auto', **kwargs)
  return fig, ax


def normalize_batch(arr):
  axes = np.arange(1, len(arr.shape))
  vmin = np.min(arr, axis=tuple(axes))
  vmax = np.max(arr, axis=tuple(axes))
  vmax = np.expand_dims(vmax - vmin, tuple(axes))
  vmin = np.expand_dims(vmin, tuple(axes))
  return (arr - vmin) / vmax


def normalize_array(imarr, vmin=None, vmax=None):
  if vmin is None:
    vmin = imarr.min()
  if vmax is None:
    vmax = imarr.max()
  return (imarr - vmin) / (vmax - vmin)