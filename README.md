# NeurIPS 2020 Flatland Challenge baselines

The basic structure of this repository is adopted from [https://github.com/spMohanty/rl-experiments/](https://github.com/spMohanty/rl-experiments/)

## Installation

Tested with Python 3.6 and 3.7

```
conda create --name ray-env python=3.7 --yes
```

You may need to install/update bazel: https://docs.bazel.build/versions/master/install-ubuntu.html

```
pip install ray[rllib]
pip install tensorflow # or tensorflow-gpu
pip install -r requirements.txt
```

## Usage
```
Training example:
    python ./train.py -f experiments/flatland_random_sparse_small/global_obs_conv_net/ppo.yaml

Test example:
    python ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run PPO --no-render
        --config '{"env_config": {"test": true}}' --episodes 1000 --out rollouts.pkl

Note that -f overrides all other trial-specific command-line options.
```

## Experiment structure

Experiments consist of one or many rllib YAML config files 
alongside a MARKDOWN file containing results, plots 
and a detailed description of the methodology.
All files are stored in a experiment folder under `experiments/<env-name>/<experiment-name>`.
An example can be found under `experiments/flatland_random_sparse_small/global_obs_conv_net/README.md`.


