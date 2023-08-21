"""score_flow tests."""
import unittest
import sys
sys.path.append('..')

import jax
from score_flow import losses
from score_flow.models import utils as mutils
from score_flow.models import ddpm, ncsnpp, ncsnv2  # pylint: disable=unused-import, g-multiple-import 

from score_configs import default_ncsnpp_config as score_model_config

# Parameters for shape of dummy data:
_IMAGE_SIZE = 8
_N_CHANNELS = 1


class ScoreFlowTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    # Set up config for DPI.
    config = score_model_config.get_config()
    config.data.image_size = _IMAGE_SIZE
    config.data.num_channels = _N_CHANNELS
    config.training.batch_size = 1
    # Reduce model size to make test faster.
    config.model.name = 'ddpm'
    config.model.nf = 32
    config.model.ch_mult = (1, 1, 1, 1)
    config.model.attn_resolutions = (1,)
    config.model.num_res_blocks = 1
    self.config = config

  def test_init_score_model(self):
    config = self.config

    # Initialize model.
    rng = jax.random.PRNGKey(config.seed)
    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, init_params = mutils.init_model(step_rng, config)

    # Initialize optimizer.
    tx = losses.get_optimizer(config)
    opt_state = tx.init(init_params)

    # Construct initial state.
    state = mutils.State(
        step=0,
        model_state=init_model_state,
        opt_state=opt_state,
        ema_rate=config.model.ema_rate,
        params=init_params,
        params_ema=init_params,
        rng=rng)
    return state, score_model, tx


if __name__ == '__main__':
  unittest.main()