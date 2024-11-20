import gymnasium as gym

from components_registry import make


class RacingEnv(gym.Env):
    def __init__(self, env_configuration: dict):
        # initialize vehicle model

        self.vehicle_model = make(
            f"dynamic_model/{env_configuration['dynamic_model']['name']}",
            vehicle_params=env_configuration["dynamic_model"]["params"],
            simulation_params=env_configuration["simulation"]["params"],
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
