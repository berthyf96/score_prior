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

"""A library of forward models with associated log-likelihood functions."""

import abc
from typing import Tuple

import ehtim as eh
import ehtim.const_def as ehc
import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
import scipy.linalg


class IndependentGaussianLikelihood(abc.ABC):
  """Likelihood module for inverse problems w/ independent Gaussian noise.
  An inverse problem with independent Gaussian noise is one whose
  forward model is given by:
    y = f(x) + noise, noise ~ N(0, diag(sigmas))
  Abstract methods should take mini-batched inputs.
  """

  @property
  @abc.abstractmethod
  def sigmas(self):
    """Std. dev. of the noise for each measured channel."""

  def invert_measurement(self,
                         y):
    """Invert measurement to a naive estimate of `x`. Optional method."""

  @abc.abstractmethod
  def apply_forward_operator(self,
                             x):
    """Apply noiseless forward operator to `x`: A*x.
    Must return measurements `y` as a flattened vector.
    Args:
      x: Source image(s).
    """

  def get_measurement(self,
                      rng,
                      x):
    """Apply forward operator to `x` and add noise."""
    # Draw Gaussian noise.
    y_dim = len(self.sigmas)
    noise = jax.random.normal(rng, (len(x), y_dim)) * self.sigmas

    y = self.apply_forward_operator(x) + noise
    return y

  def unnormalized_log_likelihood(self,
                                  x,
                                  y):
    """Unnormalized log p(y|x)."""
    residual = y - self.apply_forward_operator(x)
    log_llh = -0.5 * jnp.sum(jnp.square(residual / self.sigmas), axis=-1)
    return log_llh

  def likelihood_score(self,
                       x,
                       y):
    """Gradient of log p(y|x) with respect to `x`."""
    # `jax.grad` only takes scalar-valued functions, so we make this wrapper
    # around `self.unnormalized_log_likelihood`.
    def grad_fn(x_sample, y_sample):
      return self.unnormalized_log_likelihood(
          x_sample[None, Ellipsis], y_sample[None, Ellipsis])[0]
    return jax.vmap(jax.grad(grad_fn))(x, y)


def get_isotropic_dft_comps(n_freqs_per_orientation,
                            image_size):
  """Get the rows of the 2D DFT matrix that correspond to observed frequencies.
  Args:
    n_freqs_per_orientation: The number of lowest spatial frequencies
      that are observable either the horizontal or vertical direction.
    image_size: Height = width of the image.
  Returns:
    A 1D array containing the rows of the DFT matrix that correspond to the
      measured DFT components, assuming we can only measure up to
      `n_freqs_per_orientation` in each direction.
  """
  horizontal_dft_comps = np.arange(n_freqs_per_orientation)
  vertical_dft_comps = np.arange(n_freqs_per_orientation)
  dft_comps_2d = np.array(
      np.meshgrid(vertical_dft_comps, horizontal_dft_comps)).T.reshape(-1, 2)
  return np.ravel_multi_index(dft_comps_2d.T, (image_size, image_size))


def get_dft_matrix(image_size, dft_comps):
  """Returns the DFT operator matrix.
  Args:
    image_size: Height = width the image.
    dft_comps: A 1D array containing the indices of the rows of the full DFT
      matrix to keep.
  Returns:
    A 2D array representing the DFT measurement matrix, where only the rows
      corresponding to `dft_comps` are kept. The first half of the rows
      corresponds to the real part of the measurements, while the second
      half of the rows corresponds to the imaginary part.
  """
  dft_matrix_1d = scipy.linalg.dft(image_size)
  dft_matrix = np.kron(dft_matrix_1d, dft_matrix_1d)
  dft_matrix = dft_matrix[dft_comps]
  # Split matrix into real and imaginary submatrices.
  dft_matrix_expanded = jnp.concatenate(
      (dft_matrix.real, dft_matrix.imag), axis=0)
  return dft_matrix_expanded


class Deblurring(IndependentGaussianLikelihood):
  """Deblurring, where we observe low-frequency DFT measurements."""

  def __init__(self,
               n_freqs_per_direction,
               sigmas,
               image_shape):
    """Initialize `CompressedSensing` module.
    Args:
      n_freqs_per_direction: The number of lowest DFT components measured in
        each direction (horizontal and vertical).
      sigmas: A 1D array of the noise std. dev. for each measurement dimension.
      image_shape: The shape of one image: (image_size, image_size, n_channels).
    """
    super().__init__()
    self.n_dft = n_freqs_per_direction
    # dft_comps = get_isotropic_dft_comps(n_freqs_per_direction, image_shape[0])
    # self.dft_matrix = get_dft_matrix(image_shape[0], dft_comps)

    # Assume the noise level is the same for real and imaginary parts and for
    # each color channel. For a complex Gaussian random variable with std. dev.
    # `sigma`, the real and imaginary parts are independently Gaussian with
    # std. dev. `sigma / sqrt(2)`.
    self.real_and_imag_sigmas = jnp.tile(
        sigmas / jnp.sqrt(2), (2 * image_shape[-1]))
    self.image_shape = image_shape

  def apply_forward_operator(self, x):
    """Take subset of DFT of mini-batch `x`."""
    dft = jnp.fft.fft2(x, axes=(1, 2))
    dft = dft[:, :self.n_dft, :self.n_dft]
    measurement = dft.reshape(len(x), -1)
    measurement = jnp.concatenate((measurement.real, measurement.imag), axis=1)
    return measurement

  @property
  def sigmas(self):
    return self.real_and_imag_sigmas

  def invert_measurement(self, y):
    """Zero-fill higher DFT components and perform inverse FFT."""
    y_dim = len(self.sigmas)
    dft = y[:, :y_dim // 2] + 1j * y[:, y_dim // 2:]
    dft = dft.reshape(y.shape[0], self.n_dft, self.n_dft, -1)
    dft_zero_filled = jnp.pad(dft,
                              ((0, 0), (0, self.image_shape[0] - self.n_dft),
                               (0, self.image_shape[1] - self.n_dft), (0, 0)))
    x_recon = jnp.fft.ifft2(dft_zero_filled, axes=(1, 2))
    return x_recon.real


class Denoising(IndependentGaussianLikelihood):
  """Denoising images with iid Gaussian noise."""

  def __init__(self, scale, image_shape):
    super().__init__()
    self.scale = scale
    self.image_shape = image_shape

  @property
  def sigmas(self):
    dim = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
    return jnp.ones(dim) * self.scale

  def apply_forward_operator(self, x):
    """Identity."""
    return x.reshape(len(x), -1)

  def invert_measurement(self, y):
    """Identity."""
    return y.reshape(len(y), *self.image_shape)


class MRI(IndependentGaussianLikelihood):
  """MRI compressed sensing."""

  def __init__(self, kspace_mask: Array, sigmas: Array,
               image_shape: Tuple[int, int, int]) -> None:
    """Initialize `MRI` module.

    Args:
      kspace_mask: Mask of k-space values, an ndarray of shape
        (image_size, image_size).
      sigmas: Std. dev. of noise in k-space.
      image_shape: Image shape, i.e., (height, width, n_channels).
    """
    assert image_shape[-1] == 1  # must be grayscale
    self.kspace_mask = kspace_mask
    self.noise_sigmas = sigmas
    self.image_shape = image_shape
    # Assume the noise level is the same for real and imaginary parts.
    self.complex_sigmas = sigmas / jnp.sqrt(2) + 1j * (sigmas / jnp.sqrt(2))
    self.real_and_imag_sigmas = jnp.tile(
      sigmas / jnp.sqrt(2), (2 * image_shape[-1]))

  @property
  def sigmas(self) -> Array:
    return self.real_and_imag_sigmas

  def apply_forward_operator(self, x: Array) -> jnp.ndarray:
    """Perform FFT and then mask kspace."""
    kspace = jnp.fft.fftshift(
      jnp.fft.fftn(jnp.fft.ifftshift(
        x, axes=(1, 2)
      ), axes=(1, 2), norm='ortho'),
      axes=(1, 2)
    )
    measurement = self.kspace_mask[None, :, :, None] * kspace
    measurement = measurement.reshape(len(x), -1)
    measurement = jnp.concatenate((measurement.real, measurement.imag), axis=1)
    return measurement

  def invert_measurement(self, y: Array) -> jnp.ndarray:
    """Inverse FFT of masked kspace."""
    y_dim = len(self.sigmas)
    kspace = y[:, :y_dim // 2] + 1j * y[:, y_dim // 2:]
    kspace = kspace.reshape(y.shape[0], self.image_shape[0], self.image_shape[1], 1)
    x_recon = jnp.fft.fftshift(
      jnp.fft.ifftn(jnp.fft.ifftshift(
        kspace, axes=(1, 2)
      ), axes=(1, 2), norm='ortho'),
      axes=(1, 2)
    )
    return x_recon.real


class EHT(IndependentGaussianLikelihood):
  """EHT measurements with complex visibilities. Assumes grayscale, square images."""

  def __init__(self,
               forward_matrix,
               sigmas,
               image_size):
    """Initialize `EHT` module.
    Args:
      forward_matrix: The measurement matrix (complex-valued) for EHT
        observations.
      sigmas: The noise std. dev. (real-valued) for each measurement.
      image_size: The image height = width.
    """
    self.forward_matrix = forward_matrix
    self.forward_matrix_expanded = jnp.concatenate(
        (forward_matrix.real, forward_matrix.imag), axis=0)
    self.noise_sigmas = sigmas
    # Note: `inverse_matrix` should only be used for visualizing a naive
    # reconstruction. Since `forward_matrix` might be ill-conditioned, taking
    # the pseudo-inverse of it might not be a good idea.
    self.inverse_matrix = jnp.linalg.pinv(self.forward_matrix_expanded)
    self.image_size = image_size
    # Assume the noise level is the same for real and imaginary parts.
    self.real_and_imag_sigmas = jnp.concatenate((sigmas, sigmas))

  @property
  def sigmas(self):
    return self.real_and_imag_sigmas

  def apply_forward_operator(self, x):
    return jnp.einsum(
        'ij,bj->bi', self.forward_matrix_expanded, x.reshape(len(x), -1))

  def invert_measurement(self, y):
    x = jnp.einsum('ij,bj->bi', self.inverse_matrix, y)
    return x.reshape(len(y), self.image_size, self.image_size, 1).real


class Interferometry:
  """EHT measurements with visibility amplitudes and closure phases. Assumes grayscale, square images."""

  def __init__(self,
               obs: eh.obsdata.Obsdata,
               im: eh.image.Image,
               flux_multiplier: float,
               data_products: str = 'visamp_cphase'):
    """Initialize `Interferometry` module.
    Args:
      obs: eht-imaging `Obsdata` object. This must be preprocessed to have
        the measurements ready for inference (e.g., scaled so that source image
        has pixel values in range [0, 1], cphase and amplitude measurements
        already computed).
      im: eht-imaging `Image` object that has dummy image information.
      flux_multiplier: The multiplier that has been used to scale the
        visibilities in `obs`.
      data_products: string specifying which data products to use
        'vis' | 'visamp_cphase' | 'visamp_cphase_logcamp'.
    """
    assert im.xdim == im.ydim
    self.image_size = im.xdim
    self.obs = obs
    self.flux_multiplier = flux_multiplier
    self.data_products = data_products
    
    # Get forward model for complex visibilities.
    _, _, A_vis = eh.imaging.imager_utils.chisqdata_vis(obs, im, mask=[])

    # Get forward model for closure phases.
    _, _, A_cp = eh.imaging.imager_utils.chisqdata_cphase(obs, im, mask=[])

    # Get forward model for log-closure amplitudes.
    _, _, A_logca = eh.imaging.imager_utils.chisqdata_logcamp(obs, im, mask=[])

    self.A_vis = A_vis  # forward matrix for complex visibilities
    self.A_vis_expanded = jnp.concatenate((A_vis.real, A_vis.imag), axis=0)
    self.sigma_vis = obs.data['sigma']  # noise of visibility amplitudes
    self.sigma_vis_expanded = jnp.concatenate((self.sigma_vis, self.sigma_vis))
    self.A_cp = A_cp  # forward matrix for closure phases
    self.sigma_cp = obs.cphase['sigmacp']  # noise of closure phases
    self.A_logca = A_logca  # forward matrix for log-closure amplitudes
    self.sigma_logca = obs.logcamp['sigmaca']  # noise of log-closure amplitudes
    # Save measurements.
    self.vis = obs.data['vis']
    self.vis_expanded = jnp.concatenate((self.vis.real, self.vis.imag))
    self.visamp = obs.amp['amp']
    self.cphase = obs.cphase['cphase']
    self.logcamp = obs.logcamp['camp']
    if data_products not in ['vis', 'visamp_cphase', 'visamp_cphase_logcamp']:
      raise ValueError(f'Unrecognized data products: {data_products}')

  # @functools.partial(jax.vmap, in_axes=(None, 0))
  def apply_forward_operator(self, x):
    xvec = x.reshape(len(x), -1)

    if self.data_products == 'vis':
      # Complex visibilities
      vis_expanded = jnp.einsum('ij,bj->bi', self.A_vis_expanded, xvec)
      return vis_expanded
    
    if self.data_products in ['visamp_cphase', 'visamp_cphase_logcamp']:
      # Visibility amplitudes
      vis = jnp.einsum('ij,bj->bi', self.A_vis, xvec)
      vis_amp = jnp.abs(vis)
      # Closure phases
      i1 = jnp.einsum('ij,bj->bi', self.A_cp[0], xvec)
      i2 = jnp.einsum('ij,bj->bi', self.A_cp[1], xvec)
      i3 = jnp.einsum('ij,bj->bi', self.A_cp[2], xvec)
      cphase = jnp.angle(i1 * i2 * i3)
    if self.data_products == 'visamp_cphase':
      return vis_amp, cphase
    elif self.data_products == 'visamp_cphase_logcamp':
      # Log-closure amplitudes
      a1 = np.abs(np.dot(self.A_logca[0], xvec))
      a2 = np.abs(np.dot(self.A_logca[1], xvec))
      a3 = np.abs(np.dot(self.A_logca[2], xvec))
      a4 = np.abs(np.dot(self.A_logca[3], xvec))
      logcamp = jnp.log(a1) + jnp.log(a2) - jnp.log(a3) - jnp.log(a4)
      return vis_amp, cphase, logcamp
  
  def get_measurement(self):
    if self.data_products == 'vis':
      return self.vis_expanded[None, :]
    elif self.data_products == 'visamp_cphase':
      return self.visamp[None, :], self.cphase[None, :]
    elif self.data_products == 'visamp_cphase_logcamp':
      return self.visamp[None, :], self.cphase[None, :], self.logcamp[None, :]

  def invert_measurement(self, fov):
    dirty_image = self.obs.dirtyimage(self.image_size, fov)
    return dirty_image.imvec.reshape(1, self.image_size, self.image_size, 1)
  
  def data_fit_loss(self,
                    x,
                    vis_weight=1.,
                    visamp_weight=1.,
                    cphase_weight=1.,
                    logcamp_weight=1.):
    if self.data_products == 'vis':
      residual_expanded = self.vis_expanded[None, :] - self.apply_forward_operator(x)
      return vis_weight * jnp.mean(jnp.square(residual_expanded / self.sigma_vis_expanded), axis=-1)
    if self.data_products in ['visamp_cphase', 'visamp_cphase_logcamp']:
      visamp_pred, cphase_pred = self.apply_forward_operator(x)[:2]
      # Visibility amplitudes
      residual = self.visamp[None, :] - visamp_pred
      visamp_loss = jnp.mean(jnp.square(residual / self.sigma_vis), axis=-1)
      # Closure phases
      cphase_true = self.cphase * ehc.DEGREE
      angle_residual = cphase_true[None, :] - cphase_pred
      sigma = self.sigma_cp * ehc.DEGREE
      cphase_loss = 2. * jnp.mean((1 - jnp.cos(angle_residual)) / jnp.square(sigma), axis=-1)
    if self.data_products == 'visamp_cphase':
      return visamp_weight * visamp_loss + cphase_weight * cphase_loss
    elif self.data_products == 'visamp_cphase_logcamp':
      logcamp_pred = self.apply_forward_operator(x)[2]
      # Log-closure amplitudes
      residual = self.logcamp[None, :] - logcamp_pred
      logcamp_loss = jnp.mean(jnp.square(residual / self.sigma_logca), axis=-1)
      return visamp_weight * visamp_loss + cphase_weight * cphase_loss + logcamp_weight * logcamp_loss

  def avg_chisq_vis(self, x):
    """Compute average visibility chi^2 for a given batch of images."""
    chisq = np.zeros(len(x))
    for i, xi in enumerate(x):
      chisq[i] = eh.imaging.imager_utils.chisq_vis(
        xi.reshape(-1), self.A_vis, self.vis, self.sigma_vis)
    return np.mean(chisq)

  def avg_chisq_visamp(self, x):
    """Compute average amplitude chi^2 for a given batch of images."""
    chisq = np.zeros(len(x))
    for i, xi in enumerate(x):
      chisq[i] = eh.imaging.imager_utils.chisq_amp(
        xi.reshape(-1), self.A_vis, self.visamp, self.sigma_vis)
    return np.mean(chisq)

  def avg_chisq_cphase(self, x):
    """Compute average closure phase chi^2 for a given batch of images."""
    chisq = np.zeros(len(x))
    for i, xi in enumerate(x):
      chisq[i] = eh.imaging.imager_utils.chisq_cphase(
        xi.reshape(-1), self.A_cp, self.cphase, self.sigma_cp)
    return np.mean(chisq)
  
  def avg_chisq_logcamp(self, x):
    """Compute average log closure amplitude chi^2 for a given batch of images."""
    chisq = np.zeros(len(x))
    for i, xi in enumerate(x):
      chisq[i] = eh.imaging.imager_utils.chisq_logcamp(
        xi.reshape(-1), self.A_logca, self.logcamp, self.sigma_logca)
    return np.mean(chisq)