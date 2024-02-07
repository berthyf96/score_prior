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

"""Utility functions for posterior inference with score-based prior."""
import os

import diffrax
import ehtim as eh
import ehtim.const_def as ehc
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from score_flow.models import utils as mutils
from score_flow.models import ddpm, ncsnpp, ncsnv2  # pylint: disable=unused-import, g-multiple-import 
from sklearn.decomposition import PCA
import scipy.stats
from score_flow import datasets
from score_flow import utils
from tensorflow.io import gfile

from ehtim_utils import estimate_flux_multiplier
import forward_models
import mri
import probability_flow
from posterior_sampling import model_utils as dpi_mutils


def get_score_fn(config,
                 score_model_config):
  """Return score function for a given model checkpoint."""
  score_model_config.data = config.data
  
  if gfile.isdir(config.prob_flow.score_model_dir):
    # Try to find the latest checkpoint.
    ckpt_path = checkpoints.latest_checkpoint(config.prob_flow.score_model_dir)
  else:
    ckpt_path = config.prob_flow.score_model_dir

  if ckpt_path is None or not gfile.exists(ckpt_path):
    raise FileNotFoundError(
        'No pretrained model found in %s' % config.prob_flow.score_model_dir)

  # Initialize score model and state.
  rng = jax.random.PRNGKey(score_model_config.seed)
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, init_params = mutils.init_model(step_rng, score_model_config)

  # Construct initial state.
  state = mutils.State(
      step=0,
      model_state=init_model_state,
      opt_state=None,
      ema_rate=score_model_config.model.ema_rate,
      params=init_params,
      params_ema=init_params,
      rng=rng)
  
  # Load checkpoint.
  state = checkpoints.restore_checkpoint(ckpt_path, state)

  # Get SDE.
  sde, _ = utils.get_sde(score_model_config)
  # Get score function.
  score_fn = mutils.get_score_fn(
      sde,
      score_model,
      state.params_ema,
      state.model_state,
      train=False,
      continuous=True)
  return score_fn


def _get_solver(solver,
                scan_stages = True):
  """Return `diffrax.AbstractSolver` instance."""
  if solver == 'Euler':
    return diffrax.Euler()
  try:
    return getattr(diffrax, solver)(scan_stages=scan_stages)
  except:
    return getattr(diffrax, solver)()


def _get_stepsize_controller(stepsize_controller,
                             rtol,
                             atol,
                             adjoint_rms_seminorm = False
                             ):
  """Return `diffrax.AbstractStepSizeController` instance."""
  if stepsize_controller == 'ConstantStepSize':
    return diffrax.ConstantStepSize()
  elif stepsize_controller == 'PIDController':
    if adjoint_rms_seminorm:
      return diffrax.PIDController(
          norm=diffrax.adjoint_rms_seminorm, rtol=rtol, atol=atol)
    else:
      return diffrax.PIDController(rtol=rtol, atol=atol)
  else:
    raise ValueError(f'Unsupported stepsize controller: {stepsize_controller}')
  


def _get_adjoint_solver(adjoint_method, adjoint_solver,
                        adjoint_stepsize_controller,
                        adjoint_rtol, adjoint_atol, adjoint_rms_seminorm):
  """Return `diffrax.AbstractSolver` for the adjoint."""
  if adjoint_method == 'RecursiveCheckpointAdjoint':
    adjoint = diffrax.RecursiveCheckpointAdjoint()
  elif adjoint_method == 'BacksolveAdjoint':
    adjoint_solver = _get_solver(adjoint_solver, scan_stages=True)
    adjoint_stepsize_controller = _get_stepsize_controller(
        adjoint_stepsize_controller,
        adjoint_rtol,
        adjoint_atol,
        adjoint_rms_seminorm)
    adjoint = diffrax.BacksolveAdjoint(
        solver=adjoint_solver,
        stepsize_controller=adjoint_stepsize_controller)
  else:
    raise ValueError(
        f'Unsupported adjoint method: {adjoint_method}')
  return adjoint


def get_prob_flow(config,
                  score_model_config
                  ):
  """Return `ProbabilityFlow` module.
  Args:
    config: Config for inference setup (e.g., DPI, grad ascent). Includes
      parameters for score-model checkpoint, dataset, etc.
    score_model_config: Config for score model. Includes parameters for
      score-model architecture, SDE, etc.
  Returns:
    A `probability_flow.ProbabilityFlow` instance.
  """
  # Get SDE.
  sde, _ = utils.get_sde(score_model_config)
  # Get score function.
  score_fn = get_score_fn(config, score_model_config)

  # ODE solver and step-size controller.
  solver = _get_solver(config.prob_flow.solver, scan_stages=True)
  stepsize_controller = _get_stepsize_controller(
      config.prob_flow.stepsize_controller,
      config.prob_flow.rtol,
      config.prob_flow.atol)

  # Adjoint solver and step-size controller.
  adjoint = _get_adjoint_solver(
    config.prob_flow.adjoint_method,
    config.prob_flow.adjoint_solver,
    config.prob_flow.adjoint_stepsize_controller,
    config.prob_flow.adjoint_rtol,
    config.prob_flow.adjoint_atol,
    config.prob_flow.adjoint_rms_seminorm)

  prob_flow = probability_flow.ProbabilityFlow(
      sde=sde,
      score_fn=score_fn,
      solver=solver,
      stepsize_controller=stepsize_controller,
      adjoint=adjoint,
      n_trace_estimates=config.prob_flow.n_trace_estimates)

  return prob_flow


def _get_eht_image(config):
  image = np.load(config.likelihood.eht_image_path)
  # Rescale to [0, 1].
  return image / image.max()


def get_likelihood(config):
  """Return the likelihood module matching the config."""
  image_size = config.data.image_size
  image_shape = (
      config.data.image_size, config.data.image_size, config.data.num_channels)
  dim = np.prod(image_shape)
  noise_scale = config.likelihood.noise_scale

  if config.likelihood.likelihood == 'Denoising':
    likelihood = forward_models.Denoising(
        scale=noise_scale,
        image_shape=image_shape)
  elif config.likelihood.likelihood == 'Deblurring':
    sigmas = jnp.ones(config.likelihood.n_dft**2) * noise_scale
    likelihood = forward_models.Deblurring(
        config.likelihood.n_dft,
        sigmas=sigmas,
        image_shape=image_shape)
  elif config.likelihood.likelihood == 'EHT':
    assert config.data.num_channels == 1
    # EHT forward model matrix and noise sigmas.
    eht_matrix = np.load(config.likelihood.eht_matrix_path)
    eht_sigmas = np.load(config.likelihood.eht_sigmas_path)

    # EHT target image.
    source_image = _get_eht_image(config)

    # Multiply noise scale by flux of image.
    eht_sigmas = eht_sigmas * np.sum(source_image)
    likelihood = forward_models.EHT(eht_matrix, eht_sigmas, image_size)
  elif config.likelihood.likelihood == 'MRI':
    # NOTE: The user should make sure that `config.likelihood.noise_scale` is
    # a reasonable std. dev. for the measurement noise.
    # From DPI paper: "The k-space measurement noise is assumed Gaussian with a
    # standard deviation of 0.04% the DC (zero-frequency) amplitude."
    # Example reasonable noise scales are 9e-3 if the original 32x32 image is
    # centered in [-1, 1] and 3.6e-3 if its pixel values are between [0, 1].
    strategy = config.likelihood.mri_sampling_type
    if strategy == 'poisson':
      # Poisson sampling mask.
      seed = 0 if config.data.image_size == 128 else 1
      calib = (0, 0) if config.data.image_size == 16 else (4, 4)
      mask = mri.poisson(
          img_shape=(image_size, image_size), accel=config.likelihood.mri_accel,
          calib=calib, dtype=np.int_, seed=seed)
    elif strategy == 'cartesian':
      # Cartesian (line) mask.
      mask = mri.cartesian(
        img_shape=(image_size, image_size), accel=config.likelihood.mri_accel)
    else:
      raise ValueError(f'MRI sampling strategy {strategy} not recognized')
    likelihood = forward_models.MRI(
        mask, sigmas=jnp.ones(dim) * noise_scale, image_shape=image_shape)
  elif config.likelihood.likelihood == 'Interferometry':
    # Load simulated data.
    im = eh.image.load_fits(config.likelihood.interferometry.image_path)
    orig_flux = im.total_flux()
    im = im.regrid_image(im.fovx(), config.data.image_size)
    im.ivec = (im.ivec / np.sum(im.ivec)) * orig_flux
    obs = eh.obsdata.load_uvfits(
      config.likelihood.interferometry.obs_path, remove_nan=True)
    
    # Add non-closing systematic noise to the observation (i.e., increase error bars, not add noise to measurements)
    obs = obs.add_fractional_noise(config.likelihood.interferometry.frac_sys_noise)
    
    multiplier = estimate_flux_multiplier(
      obs,
      config.data.image_size,
      fov=config.likelihood.interferometry.fov_uas * ehc.RADPERUAS,
      zbl=config.likelihood.interferometry.zbl,
      prior_fwhm=config.likelihood.interferometry.prior_fwhm_uas * ehc.RADPERUAS)
    multiplier *= config.likelihood.interferometry.flux_multiplier_multiplier

    # # Estimate the multiplier by which to scale image pixel values to get to [0, 1] range.
    # im_blurred = im.blur_circ(obs.res())
    # # This is a rather conservative blurring, so try multiplying by 0.5-0.7.
    # # im_blurred.ivec *= 0.7
    # multiplier = round(1 / im_blurred.ivec.max()) * config.likelihood.interferometry.flux_multiplier_multiplier

    # multiplier = round(1 / im.ivec.max()) * config.likelihood.interferometry.flux_multiplier_multiplier
    print(f'multiplier: {multiplier}')

    # # Scale visibilities and visibility sigmas.
    # obs.data['vis'] *= multiplier
    # obs.data['sigma'] *= multiplier

    # # Recompute visibility amplitudes and closure quantities.
    # obs.add_amp(debias=True)
    # obs.add_cphase(count='min')
    # obs.add_logcamp(debias=True, count='min')

    likelihood = forward_models.Interferometry(
      obs, im, multiplier, config.likelihood.interferometry.diagonalize,
      config.likelihood.interferometry.add_station_sys_noise)

    # likelihood = forward_models.Interferometry(obs, im, multiplier)
  return likelihood


def get_measurement(config,
                    likelihood,
                    single_image = True
                    ):
  """Return true image and measurement.
  Args:
    config: Config for the inference module (e.g., DPI, GradientAscent).
    likelihood: Likelihood module.
    single_image: If `True`, get one image and measurement.
      If `False`, use a batch of images.
  Returns:
    image: The true image, of shape (h, w, c) if `single_image` is True,
      else (b, h, w, c).
    y: Noisy measurement, of shape (1, m) if `single_image` is True,
      else (b, m).
  """
  if config.likelihood.likelihood == 'EHT' and config.data.centered:
    raise ValueError('Do not center data for EHT likelihood.')
  if config.likelihood.likelihood == 'Interferometry':
    im = eh.image.load_fits(config.likelihood.interferometry.image_path)
    im = im.regrid_image(
      config.likelihood.interferometry.fov_uas * ehc.RADPERUAS,
      config.data.image_size)
    # im.ivec = (im.ivec / np.sum(im.ivec)) * config.likelihood.interferometry.zbl
    true_image = im.ivec.reshape(config.data.image_size, config.data.image_size, 1)
    return true_image, None
  if config.data.dataset == 'EHT':
    # 'EHT' dataset refers to one Sgr A* image taken over the course of a night.
    image = _get_eht_image(config)  # shape: (image_size, image_size)
    image = np.expand_dims(image, axis=-1)
    # Get measurement.
    x = np.expand_dims(image, axis=0)
    y = likelihood.get_measurement(jax.random.PRNGKey(0), x)
    return image, y
  if config.data.dataset == 'CelebGaussian':
    image_shape = (16, 16, 1)
    gauss_dir = '/scratch/imaging/projects/bfeng/score_prior/celeba_gauss/16x16_components=256'
    mean = np.load(os.path.join(gauss_dir, 'mean.npy'))
    components = np.load(os.path.join(gauss_dir, 'components.npy'))
    explained_variance = np.load(os.path.join(gauss_dir, 'explained_variance.npy'))
    noise_variance = np.load(os.path.join(gauss_dir, 'noise_variance.npy'))

    pca = PCA(n_components=256)
    pca.n_components_ = 256
    pca.components_ = components
    pca.explained_variance_ = explained_variance
    pca.noise_variance_ = noise_variance
    pca.mean_ = mean

    mean = pca.mean_
    cov = pca.get_covariance()
    # Pre-condition covariance matrix.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals += 0.01
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    gauss = scipy.stats.multivariate_normal(mean=mean, cov=cov, seed=config.data.shuffle_seed)
    image = gauss.rvs(size=1).reshape(image_shape)
    # Get measurement.
    x = np.expand_dims(image, axis=0)
    y = likelihood.get_measurement(jax.random.PRNGKey(0), x)
    return image, y

  data_config = ml_collections.ConfigDict()
  data_config.data = config.data
  data_config.data.random_flip = False
  data_config.eval = ml_collections.ConfigDict()
  if single_image:
    data_config.eval.batch_size = jax.device_count()
  else:
    data_config.eval.batch_size = config.eval.batch_size

  # Get true image.
  _, _, test_ds = datasets.get_dataset(
      data_config, evaluation=True, shuffle_seed=config.data.shuffle_seed,
      device_batch=False)

  if single_image:
    image = next(iter(test_ds))['image'][0].numpy()
  else:
    image = next(iter(test_ds))['image'].numpy()

  scaler = datasets.get_data_scaler(data_config)
  image = scaler(image)

  # Get measurement.
  x = np.expand_dims(image, axis=0) if single_image else image
  y = likelihood.get_measurement(jax.random.PRNGKey(0), x)

  return image, y


def get_sample_fn(dpi_config, dpi_ckpt_dir, use_train_mode=True):
  """Get sampling function of trained DPI model."""
  # Load DPI model.
  model, model_state, params = dpi_mutils.get_model_and_init_params(
    dpi_config, train=use_train_mode)

  state = dpi_mutils.State(
      step=0,
      opt_state=None,
      params=params,
      model_state=model_state,
      data_weight=1,
      prior_weight=1,
      entropy_weight=1,
      rng=jax.random.PRNGKey(dpi_config.seed + 1))

  state = checkpoints.restore_checkpoint(dpi_ckpt_dir, state)
  print(f'Found checkpoint {state.step}')

  params = state.params
  model_state = state.model_state
  sample_fn = dpi_mutils.get_sampling_fn(model, params, model_state, train=use_train_mode)
  return sample_fn, state.step
