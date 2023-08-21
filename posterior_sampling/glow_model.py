from functools import reduce
import operator
import jax
import jax.numpy as jnp
import flax.linen as nn

### From one scale to another: squeeze / unsqueeze
def squeeze(x):
  x = jnp.reshape(x, (x.shape[0], 
                      x.shape[1] // 2, 2, 
                      x.shape[2] // 2, 2,
                      x.shape[-1]))
  x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
  x = jnp.reshape(x, x.shape[:3] + (4 * x.shape[-1],))
  return x


def unsqueeze(x):
  x = jnp.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 
                      2, 2, x.shape[-1] // 4))
  x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
  x = jnp.reshape(x, (x.shape[0], 
                      2 * x.shape[1],
                      2 * x.shape[3],
                      x.shape[5]))
  return x


### From one scale to another: split / unsplit, with learnable prior
class ConvZeros(nn.Module):
  features: int
      
  @nn.compact
  def __call__(self, x, logscale_factor=3.0):
    """A simple convolutional layers initializer to all zeros"""
    x = nn.Conv(self.features, kernel_size=(3, 3),
                strides=(1, 1), padding='same',
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros)(x)
    return x


class Split(nn.Module):
  key: jax.random.PRNGKey = jax.random.PRNGKey(0)
      
  @nn.compact
  def __call__(self, x, reverse=False, z=None, eps=None, temperature=1.0):
    """Args (reverse = True):
        * z: If given, it is used instead of sampling (= deterministic mode).
            This is only used to test the reversibility of the model.
        * eps: If z is None and eps is given, then eps is assumed to be a 
            sample from N(0, 1) and rescaled by the mean and variance of 
            the prior. This is used during training to observe how sampling
            from fixed latents evolve. 
            
    If both are None, the model samples z from scratch
    """
    if not reverse:
      del z, eps, temperature
      z, x = jnp.split(x, 2, axis=-1)
        
    # Learn the prior parameters for z
    prior = ConvZeros(x.shape[-1] * 2, name="conv_prior")(x)
        
    # Reverse mode: Only return the output
    if reverse:
      # sample from N(0, 1) prior (inference)
      if z is None:
          if eps is None:
              eps = jax.random.normal(self.key, x.shape) 
          eps *= temperature
          mu, logsigma = jnp.split(prior, 2, axis=-1)
          z = eps * jnp.exp(logsigma) + mu
      return jnp.concatenate([z, x], axis=-1)
    # Forward mode: Also return the prior as it is used to compute the loss
    else:
        return z, x, prior
        
        
### Affine Coupling 
class AffineCoupling(nn.Module):
  out_dims: int
  width: int = 512
  eps: float = 1e-8
  
  @nn.compact
  def __call__(self, inputs, logdet=0, reverse=False):
    # Split
    xa, xb = jnp.split(inputs, 2, axis=-1)
    
    # NN
    net = nn.Conv(features=self.width, kernel_size=(3, 3), strides=(1, 1),
                  padding='same', name="ACL_conv_1")(xb)
    net = nn.relu(net)
    net = nn.Conv(features=self.width, kernel_size=(1, 1), strides=(1, 1),
                  padding='same', name="ACL_conv_2")(net)
    net = nn.relu(net)
    net = ConvZeros(self.out_dims, name="ACL_conv_out")(net)
    mu, logsigma = jnp.split(net, 2, axis=-1)
    # See https://github.com/openai/glow/blob/master/model.py#L376
    # sigma = jnp.exp(logsigma)
    sigma = jax.nn.sigmoid(logsigma + 2.)
    
    # Merge
    if not reverse:
      ya = sigma * xa + mu
      logdet += jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
    else:
      ya = (xa - mu) / (sigma + self.eps)
      logdet -= jnp.sum(jnp.log(sigma), axis=(1, 2, 3))
        
    y = jnp.concatenate((ya, xb), axis=-1)
    return y, logdet
    
    
### Activation Normalization
class ActNorm(nn.Module):
  scale: float = 1.
  eps: float = 1e-8

  @nn.compact
  def __call__(self, inputs, logdet=0, reverse=False):
    # Data dependent initialization. Will use the values of the batch
    # given during model.init
    axes = tuple(i for i in range(len(inputs.shape) - 1))
    def dd_mean_initializer(key, shape):
      """Data-dependant init for mu"""
      nonlocal inputs
      x_mean = jnp.mean(inputs, axis=axes, keepdims=True)
      return - x_mean
    
    def dd_stddev_initializer(key, shape):
      """Data-dependant init for sigma"""
      nonlocal inputs
      x_var = jnp.mean(inputs**2, axis=axes, keepdims=True)
      var = self.scale / (jnp.sqrt(x_var) + self.eps)
      return var
    
    # Forward
    shape = (1,) * len(axes) + (inputs.shape[-1],)
    mu = self.param('actnorm_mean', dd_mean_initializer, shape)
    sigma = self.param('actnorm_sigma', dd_stddev_initializer, shape)
    
    logsigma = jnp.log(jnp.abs(sigma))
    logdet_factor = reduce(
        operator.mul, (inputs.shape[i] for i in range(1, len(inputs.shape) - 1)), 1)
    if not reverse:
      y = sigma * (inputs + mu)
      logdet += logdet_factor * jnp.sum(logsigma)
    else:
      y = inputs / (sigma + self.eps) - mu
      logdet -= logdet_factor * jnp.sum(logsigma)
    
    # Logdet and return
    return y, logdet
    
    
### Invertible 1x1 Convolution
class Conv1x1(nn.Module):
  channels: int
  key: jax.random.PRNGKey = jax.random.PRNGKey(0)

  def setup(self):
    """Initialize P, L, U, s"""
    # W = PL(U + s)
    # Based on https://github.com/openai/glow/blob/master/model.py#L485
    c = self.channels
    # Sample random rotation matrix
    q, _ = jnp.linalg.qr(jax.random.normal(self.key, (c, c)), mode='complete')
    p, l, u = jax.scipy.linalg.lu(q)
    # Fixed Permutation (non-trainable)
    self.P = p
    self.P_inv = jax.scipy.linalg.inv(p)
    # Init value from LU decomposition
    L_init = l
    U_init = jnp.triu(u, k=1)
    s = jnp.diag(u)
    self.sign_s = jnp.sign(s)
    S_log_init = jnp.log(jnp.abs(s))
    self.l_mask = jnp.tril(jnp.ones((c, c)), k=-1)
    self.u_mask = jnp.transpose(self.l_mask)
    # Define trainable variables
    self.L = self.param("L", lambda k, sh: L_init, (c, c))
    self.U = self.param("U", lambda k, sh: U_init, (c, c))
    self.log_s = self.param("log_s", lambda k, sh: S_log_init, (c,))

  def __call__(self, inputs, logdet=0, reverse=False):
    c = self.channels
    assert c == inputs.shape[-1]
    # enforce constraints that L and U are triangular
    # in the LU decomposition
    L = self.L * self.l_mask + jnp.eye(c)
    U = self.U * self.u_mask + jnp.diag(self.sign_s * jnp.exp(self.log_s))
    logdet_factor = inputs.shape[1] * inputs.shape[2]
    
    # forward
    if not reverse:
      # lax.conv uses weird ordering: NCHW and OIHW
      W = jnp.matmul(self.P, jnp.matmul(L, U))
      y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)), 
                        W[..., None, None], (1, 1), 'same')
      y = jnp.transpose(y, (0, 2, 3, 1))
      logdet += jnp.sum(self.log_s) * logdet_factor
    # inverse
    else:
      W_inv = jnp.matmul(jax.scipy.linalg.inv(U), jnp.matmul(
          jax.scipy.linalg.inv(L), self.P_inv))
      y = jax.lax.conv(jnp.transpose(inputs, (0, 3, 1, 2)),
                        W_inv[..., None, None], (1, 1), 'same')
      y = jnp.transpose(y, (0, 2, 3, 1))
      logdet -= jnp.sum(self.log_s) * logdet_factor

    return y, logdet

### Flow
class FlowStep(nn.Module):
  nn_width: int = 512
  key: jax.random.PRNGKey = jax.random.PRNGKey(0)

  @nn.compact
  def __call__(self, x, logdet=0, reverse=False):
    out_dims = x.shape[-1]
    if not reverse:
      x, logdet = ActNorm()(x, logdet=logdet, reverse=False)
      x, logdet = Conv1x1(out_dims, self.key)(x, logdet=logdet, reverse=False)
      x, logdet = AffineCoupling(out_dims, self.nn_width)(x, logdet=logdet, reverse=False)
    else:
      x, logdet = AffineCoupling(out_dims, self.nn_width)(x, logdet=logdet, reverse=True)
      x, logdet = Conv1x1(out_dims, self.key)(x, logdet=logdet, reverse=True)
      x, logdet = ActNorm()(x, logdet=logdet, reverse=True)
    return x, logdet
    
    
### Glow model
class GLOW(nn.Module):
  K: int = 32                                       # Number of flow steps
  L: int = 3                                        # Number of scales
  nn_width: int = 512                               # NN width in Affine Coupling Layer
  learn_top_prior: bool = False                     # If true, learn prior N(mu, sigma) for zL
  key: jax.random.PRNGKey = jax.random.PRNGKey(0)
      
      
  def flows(self, x, logdet=0, reverse=False, name=""):
    """K subsequent flows. Called at each scale."""
    for k in range(self.K):
        it = k + 1 if not reverse else self.K - k
        x, logdet = FlowStep(self.nn_width, self.key, name=f"{name}/step_{it}")(
            x, logdet=logdet, reverse=reverse)
    return x, logdet
      
  
  @nn.compact
  def __call__(self, x, reverse=False, z=None, eps=None, sampling_temperature=1.0):
    """Args:
        * x: Input to the model
        * reverse: Whether to apply the model or its inverse
        * z (reverse = True): If given, use these as intermediate latents (deterministic)
        * eps (reverse = True, z!=None): If given, use these as Gaussian samples which are later 
            rescaled by the mean and variance of the appropriate prior.
        * sampling_temperature (reverse = True, z!=None): Sampling temperature
    """
    
    ## Inputs
    # Forward pass: Save priors for computing loss
    # Optionally save zs (only used for sanity check of reversibility)
    priors = []
    if not reverse:
      del z, eps, sampling_temperature
      z = []
    # In reverse mode, either use the given latent z (deterministic)
    # or sample them. For the first one, uses the top prior.
    # The intermediate latents are sampled in the `Split(reverse=True)` calls
    else:
      if z is not None:
        assert len(z) == self.L
      else:
        x *= sampling_temperature
        if self.learn_top_prior:
          # Assumes input x is a sample from N(0, 1)
          # Note: the inputs to learn the top prior is zeros (unconditioned)
          # or some conditioning e.g. class information.
          # If not learnable, the model just uses the input x directly
          # see https://github.com/openai/glow/blob/master/model.py#L109
          prior = ConvZeros(x.shape[-1] * 2, name="prior_top")(jnp.zeros(x.shape))
          mu, logsigma = jnp.split(prior, 2, axis=-1)
          x = x * jnp.exp(logsigma) + mu

    ## Multi-scale model
    logdet = 0
    for l in range(self.L):
      # Forward
      if not reverse:
        x = squeeze(x)
        x, logdet = self.flows(x, logdet=logdet,
                                reverse=False,
                                name=f"flow_scale_{l + 1}/")
        if l < self.L - 1:
            zl, x, prior = Split(
                key=self.key, name=f"flow_scale_{l + 1}/")(x, reverse=False)
        else:
            zl, prior = x, None
            if self.learn_top_prior:
                prior = ConvZeros(zl.shape[-1] * 2, name="prior_top")(jnp.zeros(zl.shape))
        z.append(zl)
        priors.append(prior)
              
      # Reverse
      else:
        if l > 0:
          x = Split(key=self.key, name=f"flow_scale_{self.L - l}/")(
              x, reverse=True, 
              z=z[-l - 1] if z is not None else None,
              eps=eps[-l - 1] if eps is not None else None,
              temperature=sampling_temperature)
        x, logdet = self.flows(
          x, logdet=logdet, reverse=True, name=f"flow_scale_{self.L - l}/")
        x = unsqueeze(x)
            
    ## Return
    return x, z, logdet, priors