#%%
import logging
import os
# from gymnasium import make
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np
from tqdm import tqdm
import torch
import yaml
import matplotlib.pyplot as plt
from track.track import Track
# change current working dir to root
os.chdir('..')

from agent import KanPPOAgent
from racing_env import RacingEnv
#%%
iter_num = 100
num_time_step = 10
dt_imposed = 0.1

track_path = "data/tracks/oval_track_20_5_8_2.npz"
track = Track.load(track_path)

kinematic_runs = [
]

run_name = kinematic_runs[-1]
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
env_conf['env_configuration']['dynamic_model']['params']["max_velocity"] = 20
env_conf['env_configuration']['dynamic_model']['name'] = "kinematic_bicycle"
env_conf['env_configuration']['track_selection']['track_controller']['params']['list_path'] = "data/track_short.yaml"
env_conf['env_configuration']['dynamics_initializer']['params']['initial_state_conditions'] = {'max_velocity_proportion': 0.5}
print(f"Rollout for {total_sim_time} seconds")


formatted_iteration_number = f"{iter_num:0{len(str(training_params['num_iterations']))}}"
model_path = f"{run_dir}/{iter_num}/ppo_continuous_action_kan_{formatted_iteration_number}.cleanrl_model"
if not os.path.exists(model_path):
    model_path = f"{run_dir}/ppo_continuous_action_kan_{formatted_iteration_number}.cleanrl_model"
if not os.path.exists(model_path):
    model_path = f"{run_dir}/{iter_num}/ppo_continuous_action_kan_"

agent = KanPPOAgent.load_model(
    model_path=model_path,
    params_path=training_params_path
)
logging.basicConfig(level=logging.DEBUG)
# env:racing_env.RacingEnv = make(training_params["env_id"], **env_conf)
env = RacingEnv(env_configuration=env_conf['env_configuration'])


for x in range(0, 2, 2):
    initial_state = {
                "x": 3,
                "y": 0,
                "heading": np.pi/2,
                "steering": 0,
                "slip_angle": 0,
                "velocity": 20,
                "acceleration": 0,
    }
    override_initial_state = True

    logged_states = [
        "x",
        "y",
        "velocity",
        "acceleration",
        "heading",
        "steering",
        "throttle",
        "braking",
        "relative_heading"
    ]
    obs, info = env.reset()
    
    if override_initial_state:
        for key, value in initial_state.items():
            env.vehicle_model.__setattr__(key, value)
        closest_index = env.track_observer.get_closest_index(
                np.array([initial_state["x"], initial_state["y"]])
            )
        env.current_progress = env.track.progress_map[closest_index]
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
    track_perception = np.empty((num_time_step+1, 37, 2))
    track_observation = env.track_observer(np.array([states["x"][-1], states["y"][-1]]), states["heading"][-1])
    track_observation = track_observation.reshape(-1, 2)
    track_perception[0] = track_observation
    perceived_index = np.zeros(
        (   num_time_step+1,
            env_conf['env_configuration']['observation_conf']['track_observer']['params']['track_description']['num_points']
            ), dtype=int
    )
    closest_index = 0
    perceived_index[0] = np.arange(
        closest_index,
        closest_index + env_conf['env_configuration']['observation_conf']['track_observer']['params']['track_description']['num_points'] * env_conf['env_configuration']['observation_conf']['track_observer']['params']['track_description']['stride'],
        env_conf['env_configuration']['observation_conf']['track_observer']['params']['track_description']['stride']
    ) % len(env.track.reference_path)
    states['relative_heading'].append(env.track_observer.get_relative_heading(np.array([states["x"][0], states["y"][0]]), states["heading"][0]))
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

        # states["throttle"].append(action[0])
        # states["braking"].append(action[1])
        
        track_observation = env.track_observer(np.array([states["x"][-1], states["y"][-1]]), states["heading"][-1])
        track_observation = track_observation.reshape(-1, 2)
        track_perception[i+1] = track_observation
        states['relative_heading'].append(env.track_observer.get_relative_heading(np.array([states["x"][-1], states["y"][-1]]), states["heading"][-1]))
        
        closest_index = env.track_observer.get_closest_index(np.array([states["x"][-1], states["y"][-1]]))
        perceived_index[i+1] = np.arange(
            closest_index,
            closest_index + env_conf['env_configuration']['observation_conf']['track_observer']['params']['track_description']['num_points'] * env_conf['env_configuration']['observation_conf']['track_observer']['params']['track_description']['stride'],
            env_conf['env_configuration']['observation_conf']['track_observer']['params']['track_description']['stride']
        ) % len(env.track.reference_path)

        if terminated:
            print("simulation ended prematurely")
            break

    # # plt.figure(figsize=(8, 8))
    # plt.plot(states["x"], states["y"], label="Vehicle Path",
    #     # marker="o", color='red'
    # )
    # plt.plot(track.reference_path[:, 0], track.reference_path[:, 1], label="Track Path")
    # plt.plot(track.left_boundaries[:, 0], track.left_boundaries[:, 1], label="Left Boundary")
    # plt.plot(track.right_boundaries[:, 0], track.right_boundaries[:, 1], label="Right Boundary")
    # plt.gca().set_aspect('equal')
    # plt.title("Vehicle Path")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # # plt.legend()
    # plt.grid()
    # plt.show()

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
    ax.set_title("Vehicle path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(track.reference_path[:, 0], track.reference_path[:, 1], label="Track Path", c='b')
    ax.plot(track.left_boundaries[:, 0], track.left_boundaries[:, 1], label="Left Boundary", c='r')
    ax.plot(track.right_boundaries[:, 0], track.right_boundaries[:, 1], label="Right Boundary", c='r')
    
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
    
    plt.plot(np.arange(len(rewards)) * dt, rewards, label='Reward')
    plt.title(f"Reward over Time")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid()
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.show()
    print(rewards)

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

# %%
pos = np.array([env.vehicle_model.x, env.vehicle_model.y])
pos = np.array([-2.5, 32.5])
heading = env.vehicle_model.heading
heading = np.pi/3
closest_index = env.track_observer.get_closest_index(pos)
track_observation = env.track_observer(pos, heading)
track_observation = track_observation.reshape(-1, 2)
track_observation = env.track_observer.get_relative_points(pos, heading, 12, 5).reshape(-1, 2)
rotation_matrix = np.array([[0, 1], [-1, 0]])
track_observation = np.dot(track_observation, rotation_matrix)
observed_reference = track_observation[0::3]
observed_left = track_observation[1::3]
observed_right = track_observation[2::3]
plt.scatter(observed_reference[:, 0], observed_reference[:, 1], c='blue', label='Reference')
plt.scatter(observed_left[:, 0], observed_left[:, 1], c='green', label='Left Boundary')
plt.scatter(observed_right[:, 0], observed_right[:, 1], c='red', label='Right Boundary')
# plt.scatter(track_observation[:, 0], track_observation[:, 1], c='blue', label='Track Observation')
plt.title("Track perception")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
# %%
num_points = env.track_observer.track_description['num_points']
stride = env.track_observer.track_description['stride']
plt.scatter(
    env.track.reference_path[closest_index:closest_index+num_points*stride:stride, 0],
    env.track.reference_path[closest_index:closest_index+num_points*stride:stride, 1],
    c='blue', label='Perceived reference'
)
plt.plot(
    env.track.reference_path[:closest_index, 0],
    env.track.reference_path[:closest_index, 1],
    c='grey', label='Unseen track'
)
plt.plot(
    env.track.reference_path[closest_index+num_points*(stride-1):, 0],
    env.track.reference_path[closest_index+num_points*(stride-1):, 1],
    c='grey',
)

plt.scatter(
    env.track.left_boundaries[closest_index:closest_index+num_points*stride:stride, 0],
    env.track.left_boundaries[closest_index:closest_index+num_points*stride:stride, 1],
    c='green', label='Perceived left boundary'
)
plt.plot(
    env.track.left_boundaries[:closest_index, 0],
    env.track.left_boundaries[:closest_index, 1],
    c='grey'
)
plt.plot(
    env.track.left_boundaries[closest_index+num_points*(stride-1):, 0],
    env.track.left_boundaries[closest_index+num_points*(stride-1):, 1],
    c='grey'
)

plt.scatter(
    env.track.right_boundaries[closest_index:closest_index+num_points*stride:stride, 0],
    env.track.right_boundaries[closest_index:closest_index+num_points*stride:stride, 1],
    c='red', label='Perceived right boundary'
)
plt.plot(
    env.track.right_boundaries[:closest_index, 0],
    env.track.right_boundaries[:closest_index, 1],
    c='grey'
)
plt.plot(
    env.track.right_boundaries[closest_index+num_points*(stride-1):, 0],
    env.track.right_boundaries[closest_index+num_points*(stride-1):, 1],
    c='grey'
)

# Plot the cyan unit vector using the heading and the pos
plt.arrow(
    pos[0], pos[1],
    np.cos(heading)*3, np.sin(heading)*3,
    head_width=1, head_length=1, fc='cyan', ec='cyan',
)

plt.gcf().set_size_inches(8, 8)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.gca().set_aspect('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.plot()
#%%
fig = plt.figure(layout="constrained", figsize=(16, 9))
gspace = fig.add_gridspec(4, 3, height_ratios = [2, 2, 1, 1])
ax1 = fig.add_subplot(gspace[:2, 0]) # track
ax2 = fig.add_subplot(gspace[:2, 1]) # perception
ax3 = fig.add_subplot(gspace[0, 2]) # steering
ax4 = fig.add_subplot(gspace[1, 2]) # relative heading
ax5 = fig.add_subplot(gspace[2:4, :2]) # velocity
ax6 = fig.add_subplot(gspace[2:4, 2:]) # acceleration

metadata = dict(title='Hot lap', artist='Racing Env',)
writer = FFMpegWriter(fps=10, metadata=metadata)
# writer = PillowWriter(fps=10, metadata=metadata)

recording_name = f"recording/{run_name}.{'mp4' if isinstance(writer, FFMpegWriter) else 'gif'}"

with writer.saving(fig, recording_name, 100):
    for i in range(len(observations)):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        
        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-1, 10)
        ax4.set_xlim(0, total_sim_time)
        ax5.set_xlim(0, total_sim_time)
        ax6.set_xlim(0, total_sim_time)
        ax4.set_ylim(-np.pi, np.pi)
        ax5.set_ylim(0, np.max(states["velocity"])+1.5)
        ax6.set_ylim(np.min(states["acceleration"])-1.5, np.max(states["acceleration"])+1.5)
        
        ax1.plot(track.reference_path[:, 0], track.reference_path[:, 1], c='blue')
        ax1.plot(track.left_boundaries[:, 0], track.left_boundaries[:, 1], c='red')
        ax1.plot(track.right_boundaries[:, 0], track.right_boundaries[:, 1], c='red')
        ax1.plot(states['x'][i], states['y'][i], 'co')
        ax1.arrow(states['x'][i], states['y'][i], 
            2 * np.cos(states['heading'][(i+1)%len(states['heading'])]), 
            2 * np.sin(states['heading'][(i+1)%len(states['heading'])]), 
            head_width=1, head_length=1, fc='c', ec='c')
        ax1.scatter(track.reference_path[perceived_index[i], 0], track.reference_path[perceived_index[i], 1], c='blue')
        ax1.scatter(track.left_boundaries[perceived_index[i], 0], track.left_boundaries[perceived_index[i], 1], c='red')
        ax1.scatter(track.right_boundaries[perceived_index[i], 0], track.right_boundaries[perceived_index[i], 1], c='red')
        ax1.set_title("Top view of the track")
        perceived_track = track_perception[i] 
        perceived_track = np.dot(perceived_track, np.array([[0, 1], [-1, 0]]))
        ax2.scatter(perceived_track[1::3, 0], perceived_track[1::3, 1], c='blue')
        ax2.scatter(perceived_track[2::3, 0], perceived_track[2::3, 1], c='red')
        ax2.scatter(perceived_track[3::3, 0], perceived_track[3::3, 1], c='red')
        ax2.arrow(
            0, 0,
            0, 1,
            fc='cyan', ec='cyan', head_width=1, head_length=1
        )
        ax2.set_title("Perceived track in the agent's frame")
        ax2.grid()
        # ax3.plot(
        #     np.arange(len(states['steering'][:i+1])) * dt,
        #     states['steering'][:i+1],
        #     c="black", label="Steering")
        # ax3.set_xlim(0, total_sim_time)
        ax3.set_xlim(-1,1)
        ax3.set_ylim(0, 1.2)
        ax3.set_title("Steering")
        circle = plt.Circle((0, 0), 1, color='black', fill=False)
        ax3.add_patch(circle)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.arrow(0, 0, np.cos(states['steering'][i] + (np.pi / 2)), np.sin(states['steering'][i] + (np.pi / 2)), head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax3.grid()
        ax4.plot(np.arange(len(states['relative_heading'][:i+1])) * dt, states['relative_heading'][:i+1], c="black", label="Relative Heading")
        ax4.set_title("Relative heading")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Relative Heading [rad]")
        ax4.grid()
        ax5.plot(np.arange(len(states["velocity"][:i+1])) * dt, states["velocity"][:i+1], c="black",label="Velocity")
        ax5.set_title("Velocity")
        ax5.set_xlabel("Time [s]")
        ax5.set_ylabel("Velocity")
        ax5.grid()
        ax6.plot(np.arange(len(states["acceleration"][:i+1])) * dt, states["acceleration"][:i+1], c="black",label="Acceleration")
        ax6.set_title("Acceleration")
        ax6.set_xlabel("Time [s]")
        ax6.set_ylabel("Acceleration")
        ax6.grid()
        writer.grab_frame()
# %%

some_index = 50
plus_index = 25
old_closest_center = env.track.center_line[some_index]
new_closest_center = env.track.center_line[some_index+plus_index//2]
old_pos = old_closest_center + np.array([-0.4, -0.1])
new_pos = new_closest_center + np.array([0.5, -1.5])
restricted_center = env.track.center_line[some_index-5: some_index+plus_index]
restricted_left = env.track.left_boundaries[some_index-5: some_index+plus_index]
restricted_right = env.track.right_boundaries[some_index-5: some_index+plus_index]

plt.plot(restricted_center[:, 0], restricted_center[:, 1], label="Center Line", marker='o', c='grey')
plt.plot(restricted_left[:, 0], restricted_left[:, 1], label="Left Boundary", marker='x', c='grey')
plt.plot(restricted_right[:, 0], restricted_right[:, 1], label="Right Boundary", marker='x', c='grey')
plt.plot(old_pos[0], old_pos[1], 'ro', label="Old Position")
plt.plot(new_pos[0], new_pos[1], 'go', label="New Position", c='c')
plt.plot(old_closest_center[0], old_closest_center[1], 'rx', label="Old Closest Center")
plt.plot(new_closest_center[0], new_closest_center[1], 'cx', label="New Closest Center", c='c')
plt.plot(
    env.track.center_line[some_index:some_index+(plus_index//2)+1, 0],
    env.track.center_line[some_index:some_index+(plus_index//2)+1, 1],
    label="Progress", c='yellow'
)
# plt.gca().set_aspect('equal')
plt.title("Progress measurement")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# %%
pos_1 = np.array(
    [states['x'][5],
    states['y'][5]]
)

pos_2 = np.array(
    [states['x'][6],
    states['y'][6]]
)


index_1 = env.track_observer.get_closest_index(pos_1)
index_2 = env.track_observer.get_closest_index(pos_2)

progress_1 = env.track.progress_map[index_1]
progress_2 = env.track.progress_map[index_2]

print(f"Progress 1: {progress_1}")
print(f"Progress 2: {progress_2}")
# %%
