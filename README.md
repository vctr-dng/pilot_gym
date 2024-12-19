# Pilot Gym

Reinforcement Learning environment for autonomous racing by training a *pilote* (French for race car driver or plane operator). 

Implementation done with [gymnasium](https://gymnasium.farama.org).

## Installation

Clone the repository

```console
git clone https://github.com/vctr-dng/pilot_gym.git
```

Create a virtual environment with the necessary dependencies

### With [uv](https://docs.astral.sh/uv)

It will install *pilot gym* in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)

```console
uv sync
```

## Usage

### Configuration

The configurations files are in the [configurations folder](./configurations/)

Configure your environment with a **[DYNAMIC_MODEL]**_conf.yaml such as [kinematic_bicycle_conf.yaml](./configurations/kinematic_bicycle_conf.yaml)

Configure your experiment with a training_**[DYNAMIC_MODEL]**_**[AGENT_TYPE]**_conf.yaml suc has [training_kinematic_kan_ppo_conf.yaml](./configurations/training_kinematic_kan_ppo_conf.yaml)

Run the training script and pass the experiment parameter files as a *--params--* argument
Make sure to have the virtual environment enabled

```console
python trainer/kan_ppo.py --params configurations/training_kinematic_kan_ppo_conf.yaml
```

## Contributing

Contribution is highly appreciated

[ruff](https://astral.sh/ruff) is used for both linting and formatting as a pre-commit hook

[uv](https://docs.astral.sh/uv/) is used for project and package management


## Acknowledgements

Currently integrates:

- [pykan](https://github.com/KindXiaoming/pykan)
- [Kolmogorov-PPO](https://github.com/victorkich/Kolmogorov-PPO)
