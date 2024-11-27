from gymnasium import register

import dynamics_initializer
import dynamics_model
import reward_model
import state_observer
import track_controller
import track_observer
import track_sampler

from .racing_env import RacingEnv as RacingEnv

register(
    id="pilot_gym/RacingEnv-v0",
    entry_point=RacingEnv,
)
