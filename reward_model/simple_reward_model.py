import numpy as np

from components_registry import register

from .base_reward_model import BaseRewardModel


@register("reward_model/simple_reward")
class SimpleRewardModel(BaseRewardModel):
    def __init__(self, coefficients: dict):
        super().__init__(coefficients)

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

        reward = progress_term + out_track_term

        reward_info = {
            "progress": progress_term,
            "out_track": out_track_term,
        }

        return reward, reward_info
