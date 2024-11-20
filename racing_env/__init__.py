from gymnasium import register

import dynamics_model
import track_observer

from .racing_env import RacingEnv as RacingEnv

register(
    id="pilot_gym/RacingEnv-v0",
    entry_point="RacingEnv",
)
