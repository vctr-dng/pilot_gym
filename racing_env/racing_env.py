import gymnasium as gym
import numpy as np


class RacingEnv(gym.Env):
    def __init__(self, env_configuration: dict):
        # initialize vehicle model

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
