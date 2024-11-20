import gymnasium as gym

from components_registry import make


class RacingEnv(gym.Env):
    def __init__(self, env_configuration: dict):
        # initialize vehicle model
        observation_conf = env_configuration["observation_conf"]

        self.vehicle_model = make(
            f"dynamic_model/{env_configuration['dynamic_model']['name']}",
            vehicle_params=env_configuration["dynamic_model"]["params"],
            simulation_params=env_configuration["simulation"]["params"],
        )

        self.state_observer = make(
            f"state_observer/{observation_conf['state_observer']['name']}",
            dynamic_model=self.vehicle_model,
            observed_state=observation_conf["state_observer"]["observed_state"],
        )

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
