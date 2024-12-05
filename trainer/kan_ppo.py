# %% change cwd to root of the project
# os.chdir("../")

import yaml

from rl_alg.ppo_continuous_action_kan import get_default_params, train

# %%


def compute_params(params):
    env_config_path = params["env_config"]
    with open(f"{env_config_path}", "r") as file:
        env_config = yaml.safe_load(file)
    simulation_params = env_config["simulation"]["params"]
    params["num_steps"] = int(
        simulation_params["time_limit"] // simulation_params["dt"]
    )

    params["batch_size"] = int(params["num_envs"] * params["num_steps"])
    params["minibatch_size"] = int(params["batch_size"] // params["num_minibatches"])

    # manually set
    # params["cuda"] = True
    # params["num_envs"] = 8

    return params


# %%
params_path = "configurations/training_kinematic_kan_ppo_conf.yaml"

if not params_path:
    params = get_default_params()
else:
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

params = compute_params(params)
print(params)

train(params)
