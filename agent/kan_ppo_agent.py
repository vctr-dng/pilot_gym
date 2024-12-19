import os
import sys

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../pykan")))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from pykan.kan import KAN


class KanPPOAgent(nn.Module):
    actor_type = "kan"
    critic_type = "kan"

    def __init__(
        self,
        obs_space_size,
        action_space_size,
        k,
        g,
        critic_hidden_sizes=[],
        actor_hidden_sizes=[],
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        critic_layer_sizes = [obs_space_size] + critic_hidden_sizes + [1]
        actor_layer_sizes = [obs_space_size] + actor_hidden_sizes + [action_space_size]

        if self.critic_type == "kan":
            width = critic_layer_sizes
            self.critic = KAN(width=width, grid=g, k=k, device=self.device)
        elif self.critic_type == "mlp":
            # self.critic = layer_init(nn.Linear(init_layer_size, 1), std=1.0)
            # self.critic = nn.Sequential(
            # layer_init(nn.Linear(np.array(
            #     obs_space_size).prod(), 64)
            # ),
            # nn.Tanh(),
            # layer_init(nn.Linear(64, 64)),
            # nn.Tanh(),
            # layer_init(nn.Linear(64, 1), std=1.0),
            # )
            self.critic = self.create_mlp(critic_layer_sizes[:-1])
            self.critic.append(
                layer_init(
                    nn.Linear(critic_layer_sizes[-2], critic_layer_sizes[-1]).to(
                        self.device
                    ),
                    std=1.0,
                )
            )
        else:
            raise ValueError("Invalid critic type")

        if self.actor_type == "kan":
            width = actor_layer_sizes
            self.actor_mean = KAN(width=width, grid=g, k=k, device=device)
        elif self.actor_type == "mlp":
            self.actor_mean = self.create_mlp(actor_layer_sizes[:-1])
            self.actor_mean.append(
                layer_init(
                    nn.Linear(actor_layer_sizes[-2], actor_layer_sizes[-1]).to(
                        self.device
                    ),
                    std=0.01,
                )
            )
        else:
            raise ValueError("Invalid actor type")
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space_size).to(device))

    def create_mlp(self, layer_sizes) -> nn.Sequential:
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(
                layer_init(
                    nn.Linear(layer_sizes[i], layer_sizes[i + 1]).to(self.device)
                )
            )
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def create_kan(self, width, grid, k, device):
        return KAN(width=width, grid=grid, k=k, device=device)

    def get_value(self, x):
        return self.critic.forward(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        # print("x device: ", x.device)
        # if action:
        # print("action device: ", action.device)
        action_mean = self.actor_mean.forward(x)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if not deterministic:
                # action = probs.sample()
                sampled_action = probs.sample()
            else:
                sampled_action = action_mean
        else:
            sampled_action = action
        action = torch.tanh(sampled_action)

        value = self.get_value(x)
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            value,
        )

    @staticmethod
    def load_model(model_path, params_path, device=None):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = yaml.safe_load(open(params_path, "r"))

        agent = KanPPOAgent(
            params["obs_space_size"],
            params["action_space_size"],
            params["k"],
            params["grid"],
            params["critic_hidden_sizes"],
            params["actor_hidden_sizes"],
            device,
        )
        agent.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            )
        )

        return agent


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
