import numpy as np
import yaml
from gymnasium import make

from racing_env import RacingEnv

DEFAULT_CONFIG_PATH = "configurations/bicycle_conf.yaml"


def load_config(config_path):
    return yaml.safe_load(open(config_path, "r"))


def test_init_reset():
    config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r"))
    print(config)
    env = RacingEnv(config)
    env.reset()


def test_make():
    config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r"))
    make("pilot_gym/RacingEnv-v0", **{"env_configuration": config})


def test_straight_line():
    obs = []
    config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r"))
    env = RacingEnv(config)
    env.reset()
    for i in range(100):
        observation, reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0, 0.0])
        )
        obs.append(observation)
        print(i, reward)


if __name__ == "__main__":
    # test_init_reset()
    # test_make()
    test_straight_line()
