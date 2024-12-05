# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from racing_env.racing_env import RacingEnv

DEFAULT_CONFIG_PATH = "configurations/bicycle_conf.yaml"
config = yaml.safe_load(open(DEFAULT_CONFIG_PATH, "r"))
# %%
log = []
env = RacingEnv(config)
env.reset()

env.vehicle_model.x = 0.0
env.vehicle_model.y = 0.0
env.vehicle_model.heading = np.pi / 4
env.vehicle_model.velocity = 5

# %%
for i in range(100):
    observation, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0, 0.0])
    )
    log.append(
        {
            "obs": observation,
            "reward": reward,
            "info": info,
            "terminated": terminated,
            "truncated": truncated,
        }
    )
    print(i, reward)

df = pd.DataFrame(log)
# %%
observed_dynamic_states = config["observation_conf"]["state_observer"]["observed_state"]

additional_track_state_name = config["observation_conf"]["track_observer"]["params"][
    "observed_state"
]
track_description = config["observation_conf"]["track_observer"]["params"][
    "track_description"
]

track_total_state = (
    len(additional_track_state_name) + track_description["num_points"] * 2 * 3
)
obs_size = len(observed_dynamic_states) + track_total_state
print(obs_size)
print(df["obs"][0].shape)
print(obs_size == df["obs"][0].shape[0])
# %%
df["reward"][0]
df["info"][0]


# %%
def unpack_obs(obs):
    unpacked_obs = dict()
    dynamic_state = obs[: len(observed_dynamic_states)]
    additional_track_state = obs[
        len(observed_dynamic_states) : len(observed_dynamic_states)
        + len(additional_track_state_name)
    ]
    observed_track_points = obs[
        len(observed_dynamic_states) + len(additional_track_state_name) :
    ]

    for i, key in enumerate(observed_dynamic_states.keys()):
        unpacked_obs[key] = dynamic_state[i]

    for i, key in enumerate(additional_track_state_name):
        unpacked_obs[key] = additional_track_state[i]

    unpacked_obs["observed_points"] = observed_track_points

    return unpacked_obs


unpack_obs(df["obs"][0])
# %%

unpacked_data = []

for i in range(df.shape[0]):
    unpacked_data.append(unpack_obs(df["obs"][i]))

unpacked_df = pd.DataFrame(unpacked_data)
# %%
unpacked_df
# %%
obs_points = unpacked_df["observed_points"][0]
obs_points
# %%
obs_points = np.array(obs_points).reshape(-1, 2)
unpacked_points = {
    "reference": [],
    "left": [],
    "right": [],
}
for i in range(0, track_description["num_points"]):
    unpacked_points["reference"].append(obs_points[i * 3])
    unpacked_points["left"].append(obs_points[i * 3 + 1])
    unpacked_points["right"].append(obs_points[i * 3 + 2])

unpacked_points["reference"] = np.array(unpacked_points["reference"])
unpacked_points["left"] = np.array(unpacked_points["left"])
unpacked_points["right"] = np.array(unpacked_points["right"])

# plot the track

plt.plot(unpacked_points["reference"][:, 0], unpacked_points["reference"][:, 1])
plt.plot(unpacked_points["left"][:, 0], unpacked_points["left"][:, 1])
plt.plot(unpacked_points["right"][:, 0], unpacked_points["right"][:, 1])
plt.legend(
    [
        "reference",
        "left",
        "right",
    ]
)
plt.show()


# %%
