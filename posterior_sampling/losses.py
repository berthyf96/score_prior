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

"""Loss functions and training step function for DPI."""
import functools

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import optax

from posterior_sampling import model_utils


def clip_grad(grad, grad_clip=1.):
  grad_norm = jnp.sqrt(
      sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad)]))
  # Clip gradient.
  clipped_grad = jax.tree_map(
      lambda x: x * (grad_clip / jnp.maximum(grad_norm + 1e-6, grad_clip)),
      grad)
  return clipped_grad


def get_optimizer(config):
  if config.optim.warmup > 0:
    schedule = optax.linear_schedule(
        init_value=0, end_value=config.optim.learning_rate,
        transition_steps=config.optim.warmup)
  else:
    schedule = optax.constant_schedule(config.optim.learning_rate)
  optimizer = optax.adam(
      learning_rate=schedule,
      b1=config.optim.adam_beta1, b2=config.optim.adam_beta2,
      eps=config.optim.adam_eps)
  return optimizer


def l1_loss_fn(x):
  """L1 loss for sparsity prior.
  Args:
    x: Image batch of shape (batch, height, width, channels).
  Returns:
    L1 loss for each image in `x`, an array of shape (batch,).
  """
  return jnp.mean(jnp.abs(x), axis=(1, 2, 3))


def tsv_loss_fn(x):
  """Total squared variation loss for smoothness prior.
  Args:
    x: Image batch of shape (batch, height, width, channels).
  Returns:
    TSV loss for each image in `x`, an array of shape (batch,).
  """
  # See https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/python/ops/image_ops_impl.py#L3353.  pylint:disable=line-too-long
  pixel_dif1 = x[:, 1:, :, :] - x[:, :-1, :, :]
  pixel_dif2 = x[:, :, 1:, :] - x[:, :, :-1, :]
  tsv = (
      jnp.mean(pixel_dif1**2, axis=(1, 2, 3)) +
      jnp.mean(pixel_dif2**2, axis=(1, 2, 3)))
  return tsv


def tv_loss_fn(x):
  """Total variation loss for smoothness prior.
  Args:
    x: Image batch of shape (batch, height, width, channels).
  Returns:
    TV loss for each image in `x`, an array of shape (batch,).
  """
  # See https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/python/ops/image_ops_impl.py#L3353.  pylint:disable=line-too-long
  pixel_dif1 = x[:, 1:, :, :] - x[:, :-1, :, :]
  pixel_dif2 = x[:, :, 1:, :] - x[:, :, :-1, :]
  tv = (
      jnp.mean(jnp.abs(pixel_dif1), axis=(1, 2, 3)) +
      jnp.mean(jnp.abs(pixel_dif2), axis=(1, 2, 3)))
  return tv


def get_data_loss_fn(likelihood, y):
  """Return data-loss function.
  Args:
    likelihood: Likelihood module.
    y: Measurements, of shape `(ybatch, ydim)`, where `ybatch` is either 1 or
      the batch size of `x`.
  Returns:
    data_loss_fn: Data-loss function, which takes a batch of images and returns
      a scalar.
  """
  def data_loss_fn(x):
    log_likelihood = likelihood.unnormalized_log_likelihood(x, y)
    return -jnp.mean(log_likelihood)
  return data_loss_fn


def get_interferometry_data_loss_fn(likelihood, vis_weight, visamp_weight, cphase_weight, logcamp_weight):
  def data_loss_fn(x):
    data_losses = likelihood.data_fit_loss(
      x, vis_weight, visamp_weight, cphase_weight, logcamp_weight)
    return jnp.mean(data_losses)
  return data_loss_fn


def get_center_loss_fn(center, dim):
  X = np.concatenate([np.arange(dim).reshape((1, dim))] * dim, 0)
  Y = np.concatenate([np.arange(dim).reshape((dim, 1))] * dim, 1)
  # Add channel axis.
  X = np.expand_dims(X, axis=-1)
  Y = np.expand_dims(Y, axis=-1)
  def center_loss_fn(x):
    denom = jnp.mean(x, axis=(1, 2, 3))
    x_norm = jnp.mean(x * X, axis=(1, 2, 3)) / denom
    y_norm = jnp.mean(x * Y, axis=(1, 2, 3)) / denom
    center_losses = 0.5 * (jnp.square(x_norm - center) + jnp.square(y_norm - center))
    return jnp.mean(center_losses)
  return center_loss_fn


def get_interferometry_prior_loss_fn(
    config,
    score_fn=None,
    sde=None,
    prob_flow=None,
    t0=1e-3,
    t1=1.,
    dt0=None,
    mean=None,
    cov=None
    ):
  prior_loss_fn = get_prior_loss_fn(
    config, score_fn, sde, prob_flow, t0, t1, dt0, mean, cov)
  
  # Center loss function.
  image_size = config.data.image_size
  center_loss_fn = get_center_loss_fn(image_size / 2 - 0.5, image_size)

  # Total prior loss.
  def interferometry_prior_loss_fn(rng, x):
    prior_loss = prior_loss_fn(rng, x)
    center_loss = center_loss_fn(x)
    return prior_loss + config.optim.center_weight * center_loss

  return interferometry_prior_loss_fn


def get_prior_loss_fn(
    config,
    score_fn=None,
    sde=None,
    prob_flow=None,
    t0=1e-3,
    t1=1.,
    dt0=None,
    mean=None,
    cov=None
    ):
  """Return prior loss function according to `config`.
  Args:
    config: ConfigDict with DPI parameters.
    score_fn: Pretrained score function s(x, t). Can be `None` if not using
      probability-bound prior (i.e., `config.optim.prior` is not `dsm` or `sm`).
    sde: `score_flow.sde_lib.SDE` object. Can be `None` if not using
      probability-bound prior (i.e., `config.optim.prior` is not `dsm` or `sm`).
    prob_flow: ProbabilityFlow object. Can be `None` if not using exact
      score-based prior (i.e., `config.optim.prior` is not `ode`).
    t0: Initial time value for ProbabilityFlow ODE integration with Diffrax.
    t1: Final time value for ProbabilityFlow ODE integration with Diffrax.
    dt0: Time step size for ProbabilityFlow ODE integration with Diffrax.
    mean: Mean of Gaussian prior. Can be `None` if not using Gaussian prior.
    cov: Full covariance matrix of Gaussian prior. Can be `None` if not using
      Gaussian prior.
  Returns:
    The prior loss function, which outputs the prior loss value for a given
      RNG and batch of data.
  """
  if config.optim.prior == 'ode':
    if prob_flow is None:
      raise ValueError('Must provide `prob_flow` for ODE prior.')
    logp_fn = functools.partial(prob_flow.logp_fn, t0=t0, t1=t1, dt0=dt0)
    def prior_loss_fn(rng, x):
      log_prior = logp_fn(rng, x)
      return -jnp.mean(log_prior)
  elif config.optim.prior.lower() == 'realnvp':
    realnvp_config = config.copy_and_resolve_references()
    model, model_state, params = model_utils.get_model_and_init_params(
        realnvp_config, train=True)
    state = model_utils.State(
        step=0,
        opt_state=None,
        params=params,
        model_state=model_state,
        rng=None)
    state = checkpoints.restore_checkpoint(
      config.optim.realnvp_checkpoint, state)
    print(f'Found RealNVP checkpoint at step {state.step}')
    variables = {'params': state.params, **state.model_state}
    def prior_loss_fn(_, x):
      batch_size = x.shape[0]
      # Apply model.
      (z, logdet), _ = model.apply(
          variables, x.reshape(batch_size, -1), reverse=False,
          mutable=list(model_state.keys()))
      # Compute loss.
      log_prob_loss = -logdet + 0.5 * jnp.sum(
          jnp.square(z) + jnp.log(2 * jnp.pi), -1)
      return jnp.mean(log_prob_loss)
  elif config.optim.prior == 'l1':
    prior_loss_fn = lambda _, x: jnp.mean(l1_loss_fn(x))
  elif config.optim.prior == 'tv':
    prior_loss_fn = lambda _, x: jnp.mean(tv_loss_fn(x))
  elif config.optim.prior == 'tsv':
    prior_loss_fn = lambda _, x: jnp.mean(tsv_loss_fn(x))
  elif config.optim.prior == 'gaussian':
    def prior_loss_fn(rng, x):
      xr = x.reshape(x.shape[0], -1)
      log_prior = jax.scipy.stats.multivariate_normal.logpdf(xr, mean, cov)
      return -jnp.mean(log_prior)
  else:
    raise NotImplementedError(f'Unknown prior: {config.optim.prior}')
  return prior_loss_fn


def entropy_weight_fn(step, start_order, decay_rate, final_weight):
  """Annealing function for entropy weight, as proposed in alpha-DPI."""
  # beta_i = beta_0 * np.exp(-step / tau)
  # return max(1, beta_i)
  if step >= (start_order * decay_rate):
    return final_weight
  return max(10**(start_order - step / decay_rate), final_weight)


def data_weight_fn(step,
                   start_order,
                   decay_rate,
                   final_data_weight):
  """Annealing function for data weight, as proposed in alpha-DPI.
  Args:
    step: Current step number.
    start_order: Initial data weight is `10**(-start_order)`.
    decay_rate: Number of steps it takes to increase data weight by one
      order of magnitude.
    final_data_weight: Maximum data weight.
  Returns:
    Data weight at training step `step`.
  """
  if step >= (start_order * decay_rate):
    return final_data_weight
  return min(10**(-start_order + step / decay_rate), final_data_weight)


def sigma_annealing_fn(step, start_order, decay_rate, final_weight):
  # return start_sigma * jnp.exp(-step * decay_rate) + final_sigma
  return jnp.maximum(10**(start_order - step / decay_rate), final_weight)
  return max(10**(start_order - step / decay_rate), final_weight)


def get_train_step_fn(
    config,
    model,
    optimizer,
    data_loss_fn,
    prior_loss_fn,
    use_score_fn = False,
    score_fn = None,
    t0 = 1e-3
    ):
  """DPI train-step function that uses VJP.
  Loss functions are defined with respect to image batches. Then `jax.vjp` is
  used to compute the gradient with respect to model parameters
  (via chain rule). The gradient of the prior loss function is estimated
  by either taking the gradient through the probability flow ODE or using
  the score model to approximate the gradient at time `t0`.
  Args:
    config: `ConfigDict` with DPI parameters.
    model: `flax.linen` model.
    optimizer: `optax` optimizer.
    data_loss_fn: Function that returns the data-likelihood loss for a given
      batch of images. The loss value should be a scalar.
    prior_loss_fn: Function that returns the prior loss for a given
      (optional) RNG key and batch of images. The loss value should be a scalar.
    use_score_fn: Whether to use neural-network score model to estimate the
      prior gradient, rather than evaluating a gradient through the entire
      probability flow ODE.
    score_fn: Only used when `use_score_fn` is `True`. A function that
      applies the score-model neural network to output a score for a given batch
      of images and diffusion time.
    t0: Only used when `use_score_fn` is `True`. The diffusion time at which to
      evaluate the score.
  Returns:
    Training step function that computes loss and updates the weights of the
      RealNVP model.
  """
  per_device_batch_size = config.training.batch_size // jax.device_count()
  sampling_shape = (
      per_device_batch_size, config.data.image_size, config.data.image_size,
      config.data.num_channels)
  t0_batch = jnp.ones(per_device_batch_size) * t0

  entropy_loss_fn = lambda reverse_logdet: -jnp.mean(reverse_logdet)

  def step_fn(rng,
              state):
    """A training step.
    Args:
      rng: JAX RNG key.
      state: `flax.struct.dataclass` containing the training state.
    Returns:
      new_state: New training state.
      (loss, loss_data, loss_prior, loss_entropy): Mean losses.
      samples: The samples drawn in this step.
    """
    sample_rng = rng
    params = state.params
    model_state = state.model_state
    opt_state = state.opt_state

    def params_to_samples_fn(p):
      sample_fn = model_utils.get_sampling_fn(
          model, p, model_state, train=True)
      (samples, _), new_model_state = sample_fn(rng, sampling_shape)
      return samples, new_model_state

    def params_to_logdet_fn(p):
      sample_fn = model_utils.get_sampling_fn(
          model, p, model_state, train=True)
      (_, logdet), _ = sample_fn(rng, sampling_shape)
      return logdet

    # Get samples, along with `jax.vjp` pullback function.
    samples, vjp_fn, new_model_state = jax.vjp(
        params_to_samples_fn, params, has_aux=True)

    # Data loss value and gradient.
    loss_data, grad_data_covector = jax.value_and_grad(data_loss_fn)(samples)
    grad_data, = vjp_fn(grad_data_covector)

    # Prior loss gradient.
    if use_score_fn:
      grad_prior_covector = -score_fn(samples, t0_batch)
      loss_prior = np.NAN
    else:
      rng, logp_rng = jax.random.split(rng)
      loss_prior, grad_prior_covector = jax.value_and_grad(
          prior_loss_fn, argnums=1)(logp_rng, samples)
    grad_prior, = vjp_fn(grad_prior_covector)

    # Get logdet term, along with `jax.vjp` pullback function.
    reverse_logdet, vjp_fn = jax.vjp(params_to_logdet_fn, params)

    # Entropy loss value and gradient.
    loss_entropy, grad_entropy_covector = jax.value_and_grad(entropy_loss_fn)(
        reverse_logdet)
    grad_entropy, = vjp_fn(grad_entropy_covector)

    # Calculate total gradient.
    data_weight = state.data_weight
    prior_weight = state.prior_weight
    entropy_weight = state.entropy_weight
    grad = jax.tree_map(
        lambda a, b, c: data_weight * a + prior_weight * b + entropy_weight * c,
        grad_data, grad_prior, grad_entropy)

    # Total loss.
    loss = (
        data_weight * loss_data + prior_weight * loss_prior +
        entropy_weight * loss_entropy)

    grad = jax.lax.pmean(grad, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    loss_prior = jax.lax.pmean(loss_prior, axis_name='batch')
    loss_data = jax.lax.pmean(loss_data, axis_name='batch')
    loss_entropy = jax.lax.pmean(loss_entropy, axis_name='batch')

    # Clip gradients by global norm.
    if config.optim.grad_clip != -1:
      grad = clip_grad(grad, grad_clip=config.optim.grad_clip)

    # Apply updates.
    updates, new_opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    step = state.step + 1
    new_state = state.replace(
        step=step,
        opt_state=new_opt_state,
        params=new_params,
        model_state=new_model_state,
        rng=sample_rng)

    return new_state, (loss, loss_data, loss_prior, loss_entropy), samples

  return step_fn


def get_gaussian_train_step_fn(config, optimizer, data_loss_fn, prior_loss_fn):
  # Sampling shape.
  image_size = config.data.image_size
  image_shape = (image_size, image_size, config.data.num_channels)
  image_dim = np.prod(image_shape)
  batch_size = config.training.batch_size
  sampling_shape = (batch_size // jax.local_device_count(), *image_shape)

  @functools.partial(jax.value_and_grad, argnums=1, has_aux=True)
  def loss_fn(rng, params, prior_weight, data_weight, entropy_weight):
    # Get samples.
    sample_fn = model_utils.get_gaussian_sampling_fn(params)
    rng, step_rng = jax.random.split(rng)
    samples = sample_fn(step_rng, sampling_shape)

    # Entropy.
    var = jnp.square(params['std'])
    entropy = entropy = 0.5 * image_dim * (1 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.sum(jnp.log(var))
    entropy_loss = -entropy

    # Data loss.
    data_loss = data_loss_fn(samples)

    # Prior loss.
    rng, step_rng = jax.random.split(rng)
    prior_loss = prior_loss_fn(step_rng, samples)

    loss = data_weight * data_loss + prior_weight * prior_loss + entropy_weight * entropy_loss
    return loss, (samples, data_loss, prior_loss, entropy_loss)
  
  def step_fn(rng, state):
    rng, step_rng = jax.random.split(rng)
    (loss, (samples, data_loss, prior_loss, entropy_loss)), grad = loss_fn(
      step_rng, state.params,
      state.prior_weight, state.data_weight, state.entropy_weight)
    
    grad = jax.lax.pmean(grad, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    data_loss = jax.lax.pmean(data_loss, axis_name='batch')
    prior_loss = jax.lax.pmean(prior_loss, axis_name='batch')
    entropy_loss = jax.lax.pmean(entropy_loss, axis_name='batch')

    # Clip gradients.
    if config.optim.grad_clip >= 0:
      grad = clip_grad(grad, config.optim.grad_clip)
    
    # Apply updates.
    updates, new_opt_state = optimizer.update(grad, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    step = state.step + 1
    new_state = state.replace(
        step=step,
        opt_state=new_opt_state,
        params=new_params,
        rng=rng)

    return new_state, (loss, data_loss, prior_loss, entropy_loss), samples
  
  return step_fn