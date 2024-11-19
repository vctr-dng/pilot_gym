import dynamics_model

from .racing_env import RacingEnv as RacingEnv

dynamics_model.register_dynamic_model("bicycle", dynamics_model.BicycleModel)
