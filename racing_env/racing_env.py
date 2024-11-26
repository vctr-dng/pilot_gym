import gymnasium as gym

from components_registry import make


class RacingEnv(gym.Env):
    def __init__(self, env_configuration: dict):
        # initialize vehicle model
        observation_conf = env_configuration["observation_conf"]
        state_observer_conf = observation_conf['state_observer']
        track_observer_conf = observation_conf['track_observer']
        track_selection_conf = env_configuration["track_selection"]

        self.vehicle_model = make(
            f"dynamic_model/{env_configuration['dynamic_model']['name']}",
            vehicle_params=env_configuration["dynamic_model"]["params"],
            simulation_params=env_configuration["simulation"]["params"],
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

        self.track_observer = make(
            f"track_observer/{track_observer_conf['name']}",
            **track_observer_conf["params"],
        )
        
        self.state_names = state_observer_conf["observed_state"]
        self.observation_space = self.state_observer.observation_size
        + self.track_observer.observation_size

        self.reset()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    @staticmethod
    def from_configurations(env_configurations: dict):
        pass
