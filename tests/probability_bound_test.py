"""Tests for probability_bound."""
import unittest
import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal
from score_flow import sde_lib
from score_flow.utils import batch_mul

import probability_bound


# Parameters for shape of dummy data:
_IMAGE_SIZE = 8
_N_CHANNELS = 1
_IMAGE_DIM = _IMAGE_SIZE * _IMAGE_SIZE * _N_CHANNELS

# Prior function: N(0, sigma**2 * I)
_SIGMA = 0.5

# Tolerance for squared error between bound and exact log-prob:
_MEAN_TOL = 1.1


def get_marginal_dist_params_fn(sde):
  def marginal_dist_params(t_batch):
    """The mean coefficient and std. dev. of the marginal distribution at t."""
    all_ones = jnp.ones((t_batch.shape[0], 1))
    mean, std = sde.marginal_prob(all_ones, t_batch)
    alpha_t = jnp.mean(mean / all_ones, axis=-1)
    beta_t = std
    return alpha_t, beta_t
  return marginal_dist_params


class ProbabilityBoundTest(unittest.TestCase):
  """Tests for probability_bound."""

  def setUp(self):
    super().setUp()
    sde = sde_lib.VPSDE(beta_min=0.1, beta_max=20.)
    marginal_dist_params = get_marginal_dist_params_fn(sde)
    def score_fn(x, t_batch):
      alpha_t, beta_t = marginal_dist_params(t_batch)
      var_t = alpha_t**2 * _SIGMA**2 + beta_t**2
      return batch_mul(-1 / var_t, x)
    
    self.bound_fn = probability_bound.get_likelihood_bound_fn(
      sde, score_fn, _IMAGE_DIM, eps=1e-12, N=10000, dsm=True,
      importance_weighting=True, eps_offset=False)

  def test_bound(self):
    """Test log-probability bound.

    The bound should be exact in expectation since the score function is known.
    """
    bound_fn = jax.jit(self.bound_fn)

    rng = jax.random.PRNGKey(0)
    rng, step_rng = jax.random.split(rng)
    x = jax.random.normal(step_rng, (64, _IMAGE_DIM))

    true_logprob = multivariate_normal.logpdf(
      x, mean=np.zeros(_IMAGE_DIM), cov=np.eye(_IMAGE_DIM) * _SIGMA**2)

    n_trials = 100
    bounds = np.zeros((n_trials, x.shape[0]))
    for trial in range(n_trials):
      rng, step_rng = jax.random.split(rng)
      bounds[trial] = bound_fn(step_rng, x)

    # Check that bound is close to exact log-prob in expectation.
    bound_mean = np.mean(bounds, axis=0)
    bound_mean_sqerr = np.square(bound_mean - true_logprob)
    self.assertLess(bound_mean_sqerr.max(), _MEAN_TOL)

    # Check that bound has low relative standard deviation (for N = 10000 and 100 trials).
    bound_relstd = np.std(bounds, axis=0) / abs(bound_mean)
    self.assertLess(bound_relstd.max(), 0.07)


if __name__ == '__main__':
  unittest.main()
