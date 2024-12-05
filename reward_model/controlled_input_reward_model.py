import numpy as np

from components_registry import register

from .base_reward_model import BaseRewardModel


@register("reward_model/controlled_input_reward")
class ControlledInputRewardModel(BaseRewardModel):
    def __init__(self, coefficients: dict):
        super().__init__(coefficients)
        self.previous_action: dict = {
            "steering_rate": 0,
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
            out_track_term = -self.coefficients["out_track"] * lateral_proportion

        # promote smoother steering inputs
        steer_cost = -self.coefficients["steering_cost"] * self.input_cost(
            self.previous_action["steering_rate"],
            action["steering_rate"],
            self.coefficients["steering_cost_pow"],
        )
        throttle_cost = -self.coefficients["throttle_cost"] * self.input_cost(
            self.previous_action["throttle"],
            action["throttle"],
            self.coefficients["throttle_cost_pow"],
        )
        brake_cost = -self.coefficients["braking_cost"] * self.input_cost(
            self.previous_action["braking"],
            action["braking"],
            self.coefficients["braking_cost_pow"],
        )
        double_pedal_cost = 0
        if action["throttle"] > 0 and action["braking"] > 0:
            double_pedal_cost = -self.coefficients["double_pedal_cost"] * (
                action["throttle"] + action["braking"]
            )

        reward = (
            progress_term
            + out_track_term
            + steer_cost
            + throttle_cost
            + brake_cost
            + double_pedal_cost
        )

        self.previous_action = action

        reward_info = {
            "progress": progress_term,
            "out_track": out_track_term,
            "steering_cost": steer_cost,
            "throttle_cost": throttle_cost,
            "braking_cost": brake_cost,
        }

        return reward, reward_info

    def input_cost(self, old_action, current_action, pow):
        return np.power(np.abs(old_action - current_action), pow) / self.dt
