"""DPI default config."""
import ml_collections


def get_config():
  """Returns the default hyperparameter configuration for RealNVP / Glow."""
  config = ml_collections.ConfigDict()

  config.model = model = ml_collections.ConfigDict()
  model.bijector = 'RealNVP'
  model.n_flow = 32  # num. of flow steps in RealNVP
  model.include_softplus = False  # whether to include softplus layer for positivity
  model.batch_norm = True
  model.init_std = 0.05
  model.seqfrac = 4  # determines num. neurons in each layer

  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 64
  training.n_iters = 20000
  training.log_freq = 100
  training.snapshot_freq = 500
  training.n_saved_checkpoints = 5
  training.check_convergence = False
  training.convergence_thresh = 0.005
  training.convergence_patience = 2
  training.eval_freq = 100

  config.optim = optim = ml_collections.ConfigDict()
  optim.warmup = 5000
  optim.learning_rate = 1e-5
  optim.grad_clip = 1.
  optim.lambda_data = 1.
  optim.lambda_prior = 1.
  optim.lambda_entropy = 1.
  optim.prior = 'ode'  # ['ode', 'l1', 'tv', 'tsv', 'realnvp']
  optim.realnvp_checkpoint = ''
  optim.adam_beta1 = 0.9
  optim.adam_beta2 = 0.999
  optim.adam_eps = 1e-8
  optim.lambda_data_start_order = 0  # initial data weight = 10**(-start_order)
  optim.lambda_data_decay_steps = 1000  # num. steps to decrease data weight by one order of magnitude
  optim.dsm_nt = 1  # num. time samples to approximate DSM objective
  # For interferometry:
  optim.center_weight = 0.
  optim.vis_weight_multiplier = 0.5
  optim.visamp_weight_multiplier = 0.5
  optim.cphase_weight_multiplier = 0.5
  optim.logcamp_weight_multiplier = 0.5
  optim.flux_multiplier_multiplier = 1.

  config.likelihood = likelihood = ml_collections.ConfigDict()
  likelihood.likelihood = ''
  likelihood.noise_scale = 0.1
  likelihood.n_dft = 8
  likelihood.mri_accel = 8
  likelihood.mri_sampling_type = 'poisson'
  likelihood.eht_image_path = ''
  likelihood.eht_matrix_path = ''
  likelihood.eht_sigmas_path = ''
  likelihood.interferometry_image_path = ''
  likelihood.interferometry_obs_path = ''
  likelihood.interferometry_data_products = 'visamp_cphase'

  config.data = data = ml_collections.ConfigDict()
  data.dataset = ''
  data.image_size = 32
  data.num_channels = 1
  data.centered = False
  data.shuffle_seed = 0
  data.tfds_dir = '/scratch/imaging/projects/bfeng/tensorflow_datasets'
  data.random_flip = False
  data.category = 'church_outdoor'
  data.antialias = True
  data.taper = False
  data.taper_gaussian_blur_sigma = 2
  data.taper_frac_radius = 0.3

  config.prob_flow = prob_flow = ml_collections.ConfigDict()
  prob_flow.score_model_dir = ''
  prob_flow.n_trace_estimates = 16
  # ODE solver.
  prob_flow.solver = 'Dopri5'
  prob_flow.stepsize_controller = 'PIDController'
  prob_flow.dt0 = 0.001
  prob_flow.rtol = 1e-3  # rtol for diffrax.PIDController
  prob_flow.atol = 1e-5  # atol for diffrax.PIDController
  # Adjoint ODE solver.
  prob_flow.adjoint_method = 'BacksolveAdjoint'
  prob_flow.adjoint_solver = 'Dopri5'
  prob_flow.adjoint_stepsize_controller = 'PIDController'
  prob_flow.adjoint_rms_seminorm = True  # seminorm can reduce speed of backprop
  prob_flow.adjoint_rtol = 1e-3
  prob_flow.adjoint_atol = 1e-5

  config.gauss = gauss = ml_collections.ConfigDict()
  gauss.mean_init_scale = 0.5
  gauss.std_init_scale = 0.1

  config.seed = 42

  return config