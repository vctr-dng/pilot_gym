import os
import sys

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../pykan")))

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import copy
import random
import time
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import racing_env as racing_env
from agent.kan_ppo_agent import KanPPOAgent as Agent

DEFAULT_PARAMS = args = {
    "exp_name": os.path.basename(__file__)[: -len(".py")],
    # the name of this experiment
    "seed": 1,
    # seed of the experiment
    "torch_deterministic": True,
    # if toggled, `torch.backends.cudnn.deterministic=False`
    "cuda": False,
    # if toggled, cuda will be enabled by default
    "capture_video": False,
    # whether to capture videos of the agent performances
    # (check out `videos` folder)
    "save_model": True,
    # whether to save model into the `runs/{run_name}` folder
    # Algorithm specific arguments
    "learning_rate": 3e-4,
    # the learning rate of the optimizer
    "anneal_lr": True,
    # Toggle learning rate annealing for policy and value networks
    "gamma": 0.99,
    # the discount factor gamma
    "gae_lambda": 0.95,
    # the lambda for the general advantage estimation
    "num_minibatches": 64,
    # the number of mini-batches
    "update_epochs": 10,
    # the K epochs to update the policy
    "norm_adv": True,
    # Toggles advantages normalization
    "clip_coef": 0.2,
    # the surrogate clipping coefficient
    "clip_vloss": True,
    # Toggles whether or not to use a clipped loss for the value function,
    # as per the paper.
    "ent_coef": 0.0,
    # coefficient of the entropy
    "vf_coef": 0.5,
    # coefficient of the value function
    "max_grad_norm": 0.5,
    # the maximum norm for the gradient clipping
    "target_kl": None,
    # the target KL divergence threshold
    # to be filled in runtime
    "batch_size": 0,
    # the batch size (computed in runtime)
    "minibatch_size": 0,
    # the mini-batch size (computed in runtime)
    "num_iterations": 0,
    # the number of iterations (computed in runtime)
    # TODO: put in a training config file
    "env_id": "pilot_gym/RacingEnv-v0",
    # the id of the environment
    "env_config": "configurations/bicycle_conf.yaml",
    "total_timesteps": 1000000,
    # total timesteps of the experiments
    "num_envs": 1,
    # the number of parallel game environments
    "num_steps": 2048,
    # the number of steps to run in each environment per policy rollout
}


def get_default_params():
    return copy.deepcopy(DEFAULT_PARAMS)


def evaluate(
    agent: Agent,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    conf: dict,  # environment conf
    params: dict,  # training params
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name, gamma, conf=conf)]
    )

    num_episodes_ended = 0
    obs, _ = envs.reset()
    episodic_returns = []
    episodic_lengths = []

    while num_episodes_ended < eval_episodes:
        for i in tqdm(
            range(params["num_steps"]),
            desc=f"Evaluation ep. {num_episodes_ended} steps",
            leave=False,
        ):
            actions, _, _, _ = agent.get_action_and_value(
                torch.Tensor(obs).to(device), deterministic=True
            )
            next_obs, reward, terminated, truncated, infos = envs.step(
                actions.detach().cpu().numpy()
            )
            obs = next_obs
            if "final_info" in infos:
                for info in infos["final_info"]:
                    cumulative_reward = info[-1]["episode"]["r"]
                    episode_length = info[-1]["episode"]["l"]
                    print(
                        f"eval_episode={len(episodic_returns)}, "\
                        f"episodic_return={cumulative_reward}, "\
                        f"episodic_length={episode_length}"\
                    )
                    episodic_returns += [cumulative_reward]
                    episodic_lengths += [episode_length]
                break
        num_episodes_ended += 1

    return episodic_returns, episodic_lengths


def make_env(env_id, idx, capture_video, run_name, gamma, conf):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **conf)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def train(params: dict):
    missing_keys = []
    for key in DEFAULT_PARAMS.keys():
        if key not in params:
            missing_keys.append(key)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys: {missing_keys}")

    conf = dict()
    conf["env_configuration"] = yaml.load(
        open(params["env_config"], "r"), Loader=yaml.SafeLoader
    )

    k_values = [params["k"]]
    g_values = [params["grid"]]

    for k in k_values:
        for g in g_values:
            run_name = (
                f"{params['env_id']}__{params['exp_name']}__{params['seed']}"\
                f"__{int(time.time())}__k{k}_g{g}_MLP"
            )

            print(f"Starting experiment : {run_name}")

            writer = SummaryWriter(f"runs/{run_name}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s"
                % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
            )

            hparams = dict()

            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    hparams[key] = value
                elif isinstance(value, (list, tuple)):
                    hparams[key] = str(value)
                else:
                    hparams[key] = str(value)

            # writer.add_hparams(
            #     hparam_dict=hparams,
            #     metric_dict=dict(),
            #     run_name=run_name,
            #     global_step=0,
            # )

            device = torch.device(
                "cuda" if torch.cuda.is_available() and params["cuda"] else "cpu"
            )

            # env setup
            envs = gym.vector.SyncVectorEnv(
                [
                    make_env(
                        params["env_id"],
                        i,
                        params["capture_video"],
                        run_name,
                        params["gamma"],
                        conf=conf,
                    )
                    for i in range(params["num_envs"])
                ]
            )
            assert isinstance(
                envs.single_action_space, gym.spaces.Box
            ), "only continuous action space is supported"

            obs_space_size = int(np.prod(envs.single_observation_space.shape))
            action_space_size = int(np.prod(envs.single_action_space.shape))

            params["obs_space_size"] = obs_space_size
            params["action_space_size"] = action_space_size

            # save params and conf as yaml file
            with open(f"runs/{run_name}/training_params.yaml", "w") as file:
                yaml.dump(params, file)

            with open(f"runs/{run_name}/env_conf.yaml", "w") as file:
                yaml.dump(conf, file)

            agent = Agent(
                obs_space_size,
                action_space_size,
                k,
                g,
                params["critic_hidden_sizes"],
                params["actor_hidden_sizes"],
                device,
            ).to(device)
            optimizer = optim.Adam(
                agent.parameters(), lr=params["learning_rate"], eps=1e-5
            )
            
            # TRY NOT TO MODIFY: seeding
            random.seed(params["seed"])
            np.random.seed(params["seed"])
            torch.manual_seed(params["seed"])
            torch.backends.cudnn.deterministic = params["torch_deterministic"]

            # ALGO Logic: Storage setup
            obs = torch.zeros(
                (params["num_steps"], params["num_envs"])
                + envs.single_observation_space.shape
            ).to(device)
            actions = torch.zeros(
                (params["num_steps"], params["num_envs"])
                + envs.single_action_space.shape
            ).to(device)
            logprobs = torch.zeros((params["num_steps"], params["num_envs"])).to(device)
            rewards = torch.zeros((params["num_steps"], params["num_envs"])).to(device)
            dones = torch.zeros((params["num_steps"], params["num_envs"])).to(device)
            values = torch.zeros((params["num_steps"], params["num_envs"])).to(device)

            # TRY NOT TO MODIFY: start the game
            global_step = 0
            start_time = time.time()
            next_obs, _ = envs.reset(seed=params["seed"])
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(params["num_envs"]).to(device)

            for iteration in range(1, params["num_iterations"] + 1):
                # Annealing the rate if instructed to do so.
                if params["anneal_lr"]:
                    frac = 1.0 - (iteration - 1.0) / params["num_iterations"]
                    lrnow = frac * params["learning_rate"]
                    optimizer.param_groups[0]["lr"] = lrnow

                for step in tqdm(
                    range(0, params["num_steps"]), desc="Steps", position=0, leave=False
                ):
                    global_step += params["num_envs"]
                    obs[step] = next_obs
                    dones[step] = next_done

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(next_obs)
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, terminations, truncations, infos = envs.step(
                        action.cpu().numpy()
                    )
                    next_done = np.logical_or(terminations, truncations)
                    rewards[step] = torch.tensor(reward).to(device).view(-1)
                    next_obs, next_done = (
                        torch.Tensor(next_obs).to(device),
                        torch.Tensor(next_done).to(device),
                    )

                    if "final_info" in infos:
                        reward_mean = []
                        for info in infos["final_info"]:
                            if info and "episode" in info[-1]:
                                # print(
                                #     f"global_step={global_step},"
                                #     f"episodic_return={info[-1]['episode']['r']}"
                                # )
                                reward_mean.append(info[-1]["episode"]["r"])
                                writer.add_scalar(
                                    "charts/episodic_length",
                                    info[-1]["episode"]["l"],
                                    global_step,
                                )
                        writer.add_scalar(
                            "charts/episodic_return",
                            np.mean(reward_mean),
                            global_step,
                        )

                print(f"ITERATION {iteration}, {params['num_steps']} steps finished")

                # bootstrap value if not done
                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(params["num_steps"])):
                        if t == params["num_steps"] - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + params["gamma"] * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + params["gamma"]
                            * params["gae_lambda"]
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values

                print(f"ITERATION {iteration}, advantages and returns calculated")

                # flatten the batch
                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(params["batch_size"])
                clipfracs = []
                for epoch in tqdm(
                    range(params["update_epochs"]), desc="Epochs", position=0
                ):
                    np.random.shuffle(b_inds)
                    for start in tqdm(
                        range(0, params["batch_size"], params["minibatch_size"]),
                        desc="Minibatches",
                        position=1,
                        leave=False,
                    ):
                        end = start + params["minibatch_size"]
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [
                                ((ratio - 1.0).abs() > params["clip_coef"])
                                .float()
                                .mean()
                                .item()
                            ]

                        mb_advantages = b_advantages[mb_inds]
                        if params["norm_adv"]:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                            )

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1 - params["clip_coef"], 1 + params["clip_coef"]
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if params["clip_vloss"]:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -params["clip_coef"],
                                params["clip_coef"],
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = (
                            pg_loss
                            - params["ent_coef"] * entropy_loss
                            + v_loss * params["vf_coef"]
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            agent.parameters(), params["max_grad_norm"]
                        )
                        optimizer.step()

                    if (
                        params["target_kl"] is not None
                        and approx_kl > params["target_kl"]
                    ):
                        break

                print(f"ITERATION {iteration}, policy and value network optimized")

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(
                    "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar(
                    "losses/old_approx_kl", old_approx_kl.item(), global_step
                )
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar(
                    "losses/explained_variance", explained_var, global_step
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

                if params["save_model"]:
                    save_dir = f"runs/{run_name}/{iteration}"
                    os.makedirs(save_dir, exist_ok=True)
                    formatted_iteration_number = (
                        f"{iteration:0{len(str(params['num_iterations']))}}"
                    )
                    model_path = f"{save_dir}/{params['exp_name']}_"\
                    f"{formatted_iteration_number}.cleanrl_model"
                    torch.save(agent.state_dict(), model_path)
                    # model_info = {
                    #     'critic_hidden_sizes': params['critic_hidden_sizes'],
                    #     'actor_hidden_sizes': params['actor_hidden_sizes'],
                    #     'k': k,
                    #     'g': g,
                    # }
                    # with open(f"runs/{run_name}/{params['exp_name']}_"
                    # f"{formatted_iteration_number}.yaml", 'w') as file:
                    #     yaml.dump(model_info, file)
                    print(f"ITERATION {iteration} model saved to {model_path}")

                    episodic_returns, episodic_lengths = evaluate(
                        agent,
                        make_env,
                        params["env_id"],
                        eval_episodes=10,
                        run_name=f"{run_name}-eval",
                        device=device,
                        gamma=params["gamma"],
                        conf=conf,
                        params=params,
                    )
                    mean_return = np.mean(episodic_returns)
                    mean_length = np.mean(episodic_lengths)
                    writer.add_scalar("eval/mean_episodic_return",
                    mean_return,
                    iteration)
                    writer.add_scalar("eval/mean_episodic_length",
                    mean_length,
                    iteration)

                    # for idx, episodic_return, episodic_length in enumerate(zip(
                    # episodic_returns,
                    # episodic_lengths)):
                    #     writer.add_scalar("eval/pisodic_return", episodic_return, idx)
                    #     writer.add_scalar(
                    # "eval/episodic_length",
                    # episodic_length,
                    # idx)

                    # writer.add_hparams(
                    #     hparam_dict=hparams,
                    #     metric_dict={
                    #         "eval/mean_episodic_return": mean_return,
                    #         "eval/mean_episodic_length": mean_length,
                    #     },
                    #     global_step=iteration,
                    #     run_name="hparams",
                    # )
                    print(f"ITERATION {iteration} ended")

            envs.close()
            writer.close()
