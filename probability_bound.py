"""Functions for computing probability lower bound (ELBO)."""
import jax
import jax.numpy as jnp
import numpy as np
from score_flow import sde_lib
from score_flow.utils import batch_mul


def get_marginal_entropy_fn(sde, image_dim):
  def marginal_entropy_fn(t):
    """Returns the entropy of Gaussian marginal distribution at time t."""
    # Get the std. dev. at time t.
    _, std = sde.marginal_prob(jnp.ones((t.shape[0], 1)), t)
    var = jnp.square(std)
    entropy = 0.5 * image_dim * (1 + jnp.log(2 * jnp.pi)) + 0.5 * image_dim * jnp.log(var)
    return entropy
  return marginal_entropy_fn


def get_div_drift_fn(sde, image_dim):
  def div_drift_fn(t):
    """Returns the divergence of SDE f(x, t) with respect to x. Assumes linear drift."""
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      drift_coeff = -0.5 * (sde.beta_0 + t * (sde.beta_1 - sde.beta_0))
    elif isinstance(sde, sde_lib.VESDE):
      drift_coeff = jnp.zeros_like(t)
    else:
      raise NotImplementedError(
        f'div(f(x, t)) not implemented for SDE of type {type(sde)}')
    return drift_coeff * image_dim
  return div_drift_fn


def get_value_div_fn(fn):
  """Return both the function value and its estimated divergence via Hutchinson's trace estimator."""

  def value_div_fn(x, t, eps):
    def value_grad_fn(data):
      f = fn(data, t)
      return jnp.sum(f * eps), f
    grad_fn_eps, value = jax.grad(value_grad_fn, has_aux=True)(x)
    return value, jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

  return value_div_fn


def get_likelihood_offset_fn(sde, score_fn, eps=1e-5):
  """Create a function to compute the unbiased log-likelihood bound of a given data point.
  """

  def likelihood_offset_fn(prng, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      N: same as input
    """
    rng, step_rng = jax.random.split(prng)
    shape = data.shape

    eps_vec = jnp.full((shape[0],), eps)
    p_mean, p_std = sde.marginal_prob(data, eps_vec)
    rng, step_rng = jax.random.split(rng)
    noisy_data = p_mean + batch_mul(p_std, jax.random.normal(step_rng, shape))
    score = score_fn(noisy_data, eps_vec)

    alpha, beta = sde.marginal_prob(jnp.ones_like(data), eps_vec)
    q_mean = noisy_data / alpha + batch_mul(beta ** 2, score / alpha)
    # q_std = beta / jnp.mean(alpha, axis=(1, 2, 3))
    q_std = beta / jnp.mean(alpha)

    n_dim = np.prod(data.shape[1:])
    p_entropy = n_dim / 2. * (np.log(2 * np.pi) + 2 * jnp.log(p_std) + 1.)
    # q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * jnp.log(q_std)) + batch_mul(0.5 / (q_std ** 2),
    #                                                                             jnp.square(data - q_mean).sum(
    #                                                                               axis=(1, 2, 3)))
    q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * jnp.log(q_std)) + batch_mul(0.5 / (q_std ** 2),
                                                                                jnp.square(data - q_mean).reshape(data.shape[0], -1).sum(axis=-1))
    offset = q_recon - p_entropy
    return offset

  return likelihood_offset_fn


def get_likelihood_bound_fn(sde, score_fn, image_dim,
                            dsm=True, eps=1e-5, N=1000, importance_weighting=True,
                            eps_offset=True):
  """Create a function to compute the unbiased log-likelihood bound of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    score_fn: Score function.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    dsm: bool. Use denoising score matching bound if enabled; otherwise use sliced score matching.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.
    N: The number of time values to be sampled.
    importance_weighting: True if enable importance weighting for potential variance reduction.
    eps_offset: True if use Jensen's inequality to offset the likelihood bound due to non-zero starting time.

  Returns:
    A function that takes random states, replicated training states, and a batch of data points
      and returns the log-likelihoods in bits/dim, the latent code, and the number of function
      evaluations cost by computation.
  """
  div_drift_fn = get_div_drift_fn(sde, image_dim)

  def likelihood_bound_fn(prng, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      N: same as input
    """
    rng, step_rng = jax.random.split(prng)
    if importance_weighting:
      time_samples = sde.sample_importance_weighted_time_for_likelihood(step_rng, (N, data.shape[0]), steps=1000, eps=eps)
      Z = sde.likelihood_importance_cum_weight(sde.T, eps=eps)
    else:
      time_samples = jax.random.uniform(step_rng, (N, data.shape[0]), minval=eps, maxval=sde.T)
      Z = 1

    shape = data.shape
    if not dsm:
      raise NotImplementedError
    else:
      def scan_fn(carry, vec_time):
        rng, value = carry
        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, shape)
        mean, std = sde.marginal_prob(data, vec_time)
        noisy_data = mean + batch_mul(std, noise)

        # Score error.
        score = score_fn(noisy_data, vec_time)
        grad = batch_mul(-(noisy_data - mean), 1 / std ** 2)
        diff = score - grad
        score_error = jnp.square(diff.reshape((diff.shape[0], -1))).sum(axis=-1)

        # Gradient norm.
        grad_norm = jnp.square(grad.reshape((grad.shape[0], -1))).sum(axis=-1)

        # Drift divergence.
        drift_div = div_drift_fn(vec_time)

        # Compute the integrand.
        _, g = sde.sde(noisy_data, vec_time)
        integrand = batch_mul(g ** 2, score_error - grad_norm) - 2 * drift_div

        # Apply reweighting for importance sampling.
        if importance_weighting:
          integrand = batch_mul(std ** 2 / g ** 2 * Z, integrand)
        return (rng, value + integrand), integrand

    (rng, integral), _ = jax.lax.scan(scan_fn, (rng, jnp.zeros((shape[0],))), time_samples)
    integral = integral / N
    mean, std = sde.marginal_prob(data, jnp.ones((data.shape[0],)) * sde.T)
    rng, step_rng = jax.random.split(rng)
    noise = jax.random.normal(step_rng, shape)
    neg_prior_logp = -sde.prior_logp(mean + batch_mul(std, noise))
    nlogp = neg_prior_logp + 0.5 * integral

    if eps_offset:
      # Offset to account for not integrating exactly to 0.
      offset_fn = get_likelihood_offset_fn(sde, score_fn, eps)
      rng, step_rng = jax.random.split(rng)
      nlogp = nlogp + offset_fn(step_rng, data)

    return -nlogp

  return likelihood_bound_fn
