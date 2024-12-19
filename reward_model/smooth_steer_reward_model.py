import numpy as np

from components_registry import register

from .base_reward_model import BaseRewardModel


@register("reward_model/smooth_steer_reward")
class SmoothSteerRewardModel(BaseRewardModel):
    def __init__(self, coefficients: dict):
        super().__init__(coefficients)
        self.previous_action: dict = {
            "steering": 0,
            "throttle": 0,
            "braking": 0,
        }
        self.dt = 0.1

    def __call__(
        self,
        action: dict,
        state_observation: dict,
        track_observation: np.array,
        non_observable_states: dict,
    ) -> float:
        reward = 0

        progress = non_observable_states["progress"]
        lateral_proportion = track_observation[0]

        progress_term = self.coefficients["progress"] * progress

        out_track_term = 0
        if np.abs(lateral_proportion) > 1:
            progress_term = 0
            out_track_term = -self.coefficients["out_track"] * np.abs(lateral_proportion)

        # promote smoother steering inputs
        steer_cost = -self.coefficients["steering_cost"] * self.input_cost(
            state_observation["previous_steering"],
            state_observation["steering"],
            self.coefficients["steering_cost_pow"],
        )

        reward = (
            progress_term + out_track_term + steer_cost
        )

        self.previous_action = action

        reward_info = {
            "progress": progress_term,
            "out_track": out_track_term,
            "steering_cost": steer_cost,
        }

        return reward, reward_info

    def input_cost(self, old_action, current_action, pow):
        return np.power(np.abs(old_action - current_action), pow) / self.dt
