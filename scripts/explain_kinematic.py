#%%
import os
# from gymnasium import make
import numpy as np
from tqdm import tqdm
import torch
import yaml
import matplotlib.pyplot as plt
from track.track import Track
# change current working dir to root
os.chdir('..')

from agent import KanPPPOAgent
from racing_env import RacingEnv
import re
#%%
iter_num = 75
num_time_step = 50
dt_imposed = 0.1

track_path = "data/tracks/oval_track_20_5_8_2.npz"
track = Track.load(track_path)

# Use regex to extract values from the track_path
match = re.search(r"oval_track_(\d+)_(\d+)_(\d+)_(\d+)", track_path)
if match:
    straight_length, width, turn_radius, sample_rate = map(int, match.groups())
else:
    raise ValueError("Track path format is incorrect")

track_info = {
    "straight_length": straight_length,
    "width": width,
    "turn_radius": turn_radius,
    "sample_rate": sample_rate,
}
#%%
kinematic_runs = [
    'RacingEnv-v0__ppo_continuous_action_kan__1__1732839319__k2_g3_MLP',
    'RacingEnv-v0__ppo_continuous_action_kan__1__1732849888__k2_g3_MLP',
    'RacingEnv-v0__ppo_continuous_action_kan__1__1732859302__k2_g3_MLP',
    'RacingEnv-v0__ppo_continuous_action_kan__1__1732868457__k2_g3_MLP',
    'RacingEnv-v0__ppo_continuous_action_kan__1__1732868956__k2_g3_MLP',
]

run_name = kinematic_runs[3]
run_dir = f"runs/pilot_gym/{run_name}"
data_dir = f"{run_dir}"
env_conf_path = f"{data_dir}/env_conf.yaml"
training_params_path = f"{data_dir}/training_params.yaml"

# load the yaml files
env_conf = yaml.safe_load(open(env_conf_path, 'r'))
training_params = yaml.safe_load(open(training_params_path, 'r'))
if dt_imposed:
    env_conf['env_configuration']['simulation']['params']["dt"] = dt_imposed
dt = env_conf['env_configuration']['simulation']['params']["dt"]
total_sim_time = num_time_step*dt
env_conf['env_configuration']['simulation']['params']["time_limit"] = total_sim_time
env_conf['env_configuration']['dynamic_model']['params']["max_velocity"] = 30
env_conf['env_configuration']['dynamic_model']['name'] = "kinematic_bicycle"
env_conf['env_configuration']['track_selection']['track_controller']['params']['list_path'] = "data/track_short.yaml"
env_conf['env_configuration']['dynamics_initializer']['params']['initial_state_conditions'] = {'max_velocity_proportion': 0.5}
print(f"Rollout for {total_sim_time} seconds")

formatted_iteration_number = f"{iter_num:0{len(str(training_params['num_iterations']))}}"
model_path = f"{run_dir}/{iter_num}/ppo_continuous_action_kan_{formatted_iteration_number}.cleanrl_model"
if not os.path.exists(model_path):
    model_path = f"{run_dir}/ppo_continuous_action_kan_{formatted_iteration_number}.cleanrl_model"

agent = KanPPPOAgent.load_model(
    model_path=model_path,
    params_path=training_params_path
)

# env:racing_env.RacingEnv = make(training_params["env_id"], **env_conf)
env = RacingEnv(env_configuration=env_conf['env_configuration'])

logged_states = [
        "x",
        "y",
        "velocity",
        "heading_dot",
        "acceleration",
        # "slip_angle",
        "heading",
        "steering",
        "throttle",
        "braking",
    ]

def setup_env(env_conf, initial_state):
    env = RacingEnv(env_configuration=env_conf['env_configuration'])
    for key, value in initial_state.items():
        env.vehicle_model.__setattr__(key, value)
    return env
    
def rollout(env, agent, time):
    sim_time = 0
    n_time_steps = int(time / dt)
    actions = []
    observations = []
    rewards = []
    infos = []
    
    states = dict()
    for state in logged_states:
        states[state] = list()
    for key in states.keys():
        try:
            states[key].append(env.vehicle_model.__getattribute__(key))
        except Exception:
                states[key].append(0)
    obs, obs_info = env.get_observation()
    for i in tqdm(range(n_time_steps), desc="Rollout"):
        tensor_obs = torch.Tensor(
            obs,
        ).unsqueeze(0).to(next(agent.actor_mean.parameters()).device)
        action, _, _, _ = agent.get_action_and_value(tensor_obs, deterministic=True)
        action = action.cpu().detach().numpy().squeeze()
        # action = np.array([0, 0, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
        sim_time += dt
        
        for key in states.keys():
            try:
                states[key].append(env.vehicle_model.__getattribute__(key))
            except Exception:
                pass
        states["throttle"].append(action[0])
        states["braking"].append(action[1])
        if terminated:
            print("simulation ended prematurely")
            break
    
    return observations, actions, rewards, infos, states

initial_state = {
                "x": 0,
                "y": 0,
                "heading": np.pi/2,
                # "steering": 0,
                "velocity": 20,
                # "acceleration": 0,
}

n_points_per_side = track_info["width"]
grid_points = np.zeros((len(track.center_line), n_points_per_side * 2 + 1, 2))
local_headings = np.zeros(len(track.center_line))  
for i in range(0, len(track.center_line), track_info["sample_rate"]):
    center_point = track.center_line[i]
    left_point = track.left_boundaries[i]
    right_point = track.right_boundaries[i]
    for t in range(0, n_points_per_side):
        interp_point = left_point + t * (center_point - left_point) / n_points_per_side
        grid_points[i, t] = interp_point
    grid_points[n_points_per_side] = center_point
    for t in range(0, n_points_per_side):
        interp_point = center_point + (t + 1) * (right_point - center_point) / n_points_per_side
        grid_points[i, n_points_per_side + t + 1] = interp_point
    local_headings[i] = track.local_heading_map[i]

# plt.plot(grid_points[0,: , 0], grid_points[0,:, 1], 'o')
# plt.plot(track.center_line[:, 0], track.center_line[:, 1], label="center")
# plt.plot(track.left_boundaries[:, 0], track.left_boundaries[:, 1], label="left")
# plt.plot(track.right_boundaries[:, 0], track.right_boundaries[:, 1], label="right")
# plt.show()

mvt_field = np.zeros((len(track.center_line), n_points_per_side * 2 + 1, 2))
for index, line in enumerate(grid_points):
    initial_state = {
                "x": 0,
                "y": 0,
                "heading": np.pi/2,
                # "steering": 0,
                "velocity": 20,
                # "acceleration": 0,
    }
    initial_state["heading"] = local_headings[index]
    for j, point in enumerate(line):
        initial_state["x"] = point[0]
        initial_state["y"] = point[1]
        env = setup_env(env_conf, initial_state)
        observations, actions, rewards, infos, states = rollout(env, agent, 0.5)
        end_pos = np.array([states["x"][-1], states["y"][-1]])
        mvt_vector = end_pos - np.array([initial_state["x"], initial_state["y"]])
        mvt_field[index, j, :] = mvt_vector
    

#%%
# plot the mvt vector field
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(len(track.center_line)):
    # ax.plot(grid_points[i, :, 0], grid_points[i, :, 1], 'o')
    ax.quiver(grid_points[i, :, 0], grid_points[i, :, 1], mvt_field[i, :, 0], mvt_field[i, :, 1])

ax.plot(track.center_line[:, 0], track.center_line[:, 1], label="center")
ax.plot(track.left_boundaries[:, 0], track.left_boundaries[:, 1], label="left")
ax.plot(track.right_boundaries[:, 0], track.right_boundaries[:, 1], label="right")
ax.set_aspect('equal')
    
    

#%%



# %%
pos = np.array([env.vehicle_model.x, env.vehicle_model.y])
heading = env.vehicle_model.heading
track_observation = env.track_observer(pos, heading)
track_observation = track_observation.reshape(-1, 2)
plt.scatter(track_observation[:, 0], track_observation[:, 1], c='blue', label='Track Observation')
plt.title("Track Observation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
# %%

# car_heading = np.array(states["heading"]) % (2*np.pi)

# live_track_heading = []
# for i in range(len(states["x"])):
#     pos = np.array([
#         states["x"][i],
#         states["y"][i]
#     ])
#     closest_index = env.track_observer.get_closest_index(pos)
#     heading = env.track.local_heading_map[closest_index]
#     live_track_heading.append((heading))
# fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# # Plot vehicle and track heading
# axs[0].plot(np.arange(len(states["heading"])) * dt,
#             car_heading,
#             label="Vehicle Heading")
# axs[0].plot(np.arange(len(live_track_heading)) * dt,
#             live_track_heading, label="Track Heading")
# axs[0].set_title("Vehicle and Track Heading over Time")
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("Heading [rad]")
# axs[0].legend()
# axs[0].grid()

# # Plot heading difference
# heading_difference = (car_heading - np.array(live_track_heading))
# heading_difference = (heading_difference + np.pi) % (2 * np.pi) - np.pi
# axs[1].plot(np.arange(len(heading_difference)) * dt, heading_difference, label="Heading Difference")
# axs[1].set_title("Heading Difference over Time")
# axs[1].set_xlabel("Time [s]")
# axs[1].set_ylabel("Heading Difference [rad]")
# axs[1].legend()
# axs[1].grid()
# # Add horizontal bars at -np.pi/2 and np.pi/2 in the second subplot
# axs[1].axhline(y=-np.pi/2, color='r', linestyle='--', label='-π/2')
# axs[1].axhline(y=np.pi/2, color='g', linestyle='--', label='π/2')

# # Add legends to the plots
# axs[0].legend()
# axs[1].legend()

# plt.tight_layout()
# plt.show()

#%%
for x in range(0, 2, 2):
    initial_state = {
                "x": 0,
                "y": 0,
                "heading": np.pi/2,
                # "steering": 0,
                "velocity": 20,
                # "acceleration": 0,
    }
    initial_state["v_x"] = initial_state['velocity']
    logged_states = [
        "x",
        "y",
        # "v_x",
        # "v_y",
        "velocity",
        "heading_dot",
        # "acceleration",
        # "slip_angle",
        "heading",
        "steering",
        "throttle",
        "braking",
    ]

    obs, info = env.reset()

    for key, value in initial_state.items():
        env.vehicle_model.__setattr__(key, value)

    obs, obs_info = env.get_observation()
    observations = [obs]
    actions = []
    rewards = []
    infos = [info]
    states = dict()
    for state in logged_states:
        states[state] = list()
    
    for key in states.keys():
        try:
            states[key].append(env.vehicle_model.__getattribute__(key))
        except Exception:
                states[key].append(0)
    sim_time = 0
    for i in tqdm(range(num_time_step), desc="Rollout"):
        tensor_obs = torch.Tensor(
            obs,
        ).unsqueeze(0).to(next(agent.actor_mean.parameters()).device)
        action, _, _, _ = agent.get_action_and_value(tensor_obs, deterministic=True)
        action = action.cpu().detach().numpy().squeeze()
        # action = np.array([0, 0, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
        sim_time += dt
        
        for key in states.keys():
            try:
                states[key].append(env.vehicle_model.__getattribute__(key))
            except Exception:
                pass

        states["throttle"].append(action[0])
        states["braking"].append(action[1])
        

        if terminated:
            print("simulation ended prematurely")
            break

    # Create a colormap based on time
    time_steps = np.arange(len(states["x"]))
    norm = plt.Normalize(time_steps.min(), time_steps.max())
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(time_steps))

    # Plot the vehicle path with changing colors
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(states["x"]) - 1):
        ax.plot(states["x"][i:i+2], states["y"][i:i+2], color=colors[i])
    ax.set_aspect('equal')
    ax.set_title("Vehicle Path with Time-based Color")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(track.reference_path[:, 0], track.reference_path[:, 1], label="Track Path")
    ax.plot(track.left_boundaries[:, 0], track.left_boundaries[:, 1], label="Left Boundary")
    ax.plot(track.right_boundaries[:, 0], track.right_boundaries[:, 1], label="Right Boundary")
    
    ax.grid()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time Step')

    plt.show()
    
    for state in logged_states:
        if state in ["x", "y"]:
            continue
        plt.clf()
        plt.plot(np.arange(len(states[state])) * dt, states[state], label=state)
        plt.title(f"{state} over Time")
        plt.xlabel("Time [s]")
        plt.ylabel(state)
        plt.legend()
        plt.grid()
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)
        plt.show()
