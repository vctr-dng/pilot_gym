from typing import Optional

import gymnasium as gym
import numpy as np

from components_registry import make


class RacingEnv(gym.Env):
    def __init__(self, env_configuration: dict):
        # initialize vehicle model
        observation_conf = env_configuration["observation_conf"]
        state_observer_conf = observation_conf["state_observer"]
        track_observer_conf = observation_conf["track_observer"]
        track_selection_conf = env_configuration["track_selection"]
        self.simulation_params = env_configuration["simulation"]["params"]

        self.observed_dynamics_info = state_observer_conf["observed_state"]
        self.action_processing_info = env_configuration["action_processing"]

        self.dynamics_initializer = make(
            f"dynamics_initializer/{env_configuration['dynamics_initializer']['name']}",
            **env_configuration["dynamics_initializer"]["params"],
            vehicle_params=env_configuration["dynamic_model"]["params"],
            simulation_params=self.simulation_params,
        )

        self.vehicle_model = make(
            f"dynamic_model/{env_configuration['dynamic_model']['name']}",
            vehicle_params=env_configuration["dynamic_model"]["params"],
            simulation_params=self.simulation_params,
        )

        self.state_observer = make(
            f"state_observer/{state_observer_conf['name']}",
            dynamic_model=self.vehicle_model,
            observed_state=state_observer_conf["observed_state"],
        )

        self.track_controller = make(
            f"track_controller/{track_selection_conf['track_controller']['name']}",
            **track_selection_conf["track_controller"]["params"],
        )

        self.track_sampler = make(
            f"track_sampler/{track_selection_conf['track_sampler']['name']}",
            controller=self.track_controller,
            **track_selection_conf["track_sampler"]["params"],
        )

        self.track = self.track_sampler()

        self.track_observer = make(
            f"track_observer/{track_observer_conf['name']}",
            **track_observer_conf["params"],
            track=self.track,
        )

        self.reward_model = make(
            f"reward_model/{env_configuration['reward']['name']}",
            **env_configuration["reward"]["params"],
        )

        self.action_names = self.vehicle_model.actions
        self.action_space = gym.spaces.Box(-1, 1, (len(self.action_names),))

        self.state_names = state_observer_conf["observed_state"]
        self.observation_space_size = (
            self.state_observer.observation_size + self.track_observer.observation_size
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.observation_space_size,)
        )

        self.time_step = self.simulation_params["dt"]
        self.time_limit = self.simulation_params["time_limit"]
        self.simulation_time = 0

        self.current_progress = 0

        self.reset()

    def step(self, action: np.array):
        processed_action: dict = self.process_action(action)

        self.vehicle_model.step(processed_action)
        self.simulation_time += self.time_step

        vehicle_pos = np.array(
            [
                self.state_observer.query("x"),
                self.state_observer.query("y"),
            ]
        )

        old_progress = self.current_progress
        closest_index = self.track_observer.get_closest_index(vehicle_pos)
        self.current_progress = self.track.progress_map[closest_index]
        delta_progress = (
            self.current_progress - old_progress
        ) % self.track.progress_map[
            -1
        ]  # TODO: integrate delta_progress in track observation

        non_observable_states = {
            "progress": delta_progress,
        }

        observation, obs_info = self.get_observation()

        reward, reward_info = self.get_reward(
            processed_action,
            obs_info["state_observation_dict"],
            obs_info["track_observation"],
            non_observable_states,
        )

        terminated, truncated = self.check_episode_end()

        info = self.get_info(reward_info)

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Sample a track

        self.track = self.track_sampler()

        # Set the vehicle model to the initial state

        initial_state = self.dynamics_initializer(self.track)
        self.vehicle_model.reset(initial_state)

        observation, _ = self.get_observation()

        return observation, self.get_info({})

    def render(self):
        pass

    def close(self):
        pass

    def process_action(self, action: np.array):
        action_dic = self.action_array_to_dict(action)

        for action_name, processing_info in self.action_processing_info.items():
            action_dic[action_name] = self.singular_process(
                action_dic[action_name], processing_info
            )

        return action_dic

    def action_array_to_dict(self, action: np.array):
        processed_action: dict = dict.fromkeys(self.action_names)
        for i, action_name in enumerate(self.action_names):
            processed_action[action_name] = action[i]

        return processed_action

    def get_observation(self):
        state_observation_dict: dict = self.state_observer()

        state_observation = np.empty(self.state_observer.observation_size)
        for i, (state_name, processing_info) in enumerate(
            self.observed_dynamics_info.items()
        ):
            state_observation[i] = self.singular_process(
                state_observation_dict[state_name], processing_info
            )

        vehicle_pos = np.array(
            [
                state_observation_dict["x"],
                state_observation_dict["y"],
            ]
        )
        heading = self.vehicle_model.heading
        track_observation = self.track_observer(vehicle_pos, heading)

        observation = np.hstack((state_observation, track_observation))
        obs_info = {
            "state_observation_dict": state_observation_dict,
            "track_observation": track_observation,
        }

        return observation, obs_info

    def get_reward(
        self,
        action: dict,
        state_observation: dict,
        track_observation: np.array,
        non_observable_states: dict,
    ):
        reward, info = self.reward_model(
            action, state_observation, track_observation, non_observable_states
        )

        return reward, info

    def check_episode_end(self):
        terminated, truncated = False, False

        current_pos = np.array(
            [
                self.state_observer.query("x"),
                self.state_observer.query("y"),
            ]
        )
        current_absolute_heading = self.state_observer.query("heading")

        relative_heading = self.track_observer.get_relative_heading(
            current_pos, current_absolute_heading
        )

        if np.abs(relative_heading) >= np.pi / 2:
            terminated = True

        if (
            self.state_observer.query("velocity")
            < self.simulation_params["min_velocity"]
        ):
            terminated = True

        lateral_proportion = self.track_observer.get_lateral_proportion(current_pos)

        if np.abs(lateral_proportion) >= 1.25:
            terminated = True

        if terminated:
            return terminated, truncated

        if self.simulation_time >= self.time_limit:
            truncated = True

        return terminated, truncated

    def get_info(self, reward_info):
        base_info = dict()

        base_info["simulation_time"] = self.simulation_time
        base_info["reward_info"] = reward_info

        return base_info

    @staticmethod
    def singular_process(og_value, info):
        new_value = og_value

        if "min" and "max" in info:
            new_value = (og_value - info["min"]) / (info["max"] - info["min"])
        elif "scale" and "offset" in info:
            new_value = og_value * info["scale"] + info["offset"]

        return new_value

    @staticmethod
    def from_configurations(env_configurations: dict):
        pass
