# NeurIPS 2020 Flatland Challenge baselines

📈 [**Results**](https://app.wandb.ai/masterscrat/flatland/reports/Flatland-Baselines--Vmlldzo4OTc5NA) 

Setup
---

Tested with Python 3.6 and 3.7

```
conda create --name ray-env python=3.7 --yes
```

You may need to install/update bazel: [Ubuntu guide](https://docs.bazel.build/versions/master/install-ubuntu.html)

```
pip install ray[rllib]
pip install tensorflow # or tensorflow-gpu
pip install -r requirements.txt
```

## Usage
Training example:

`python ./train.py -f experiments/flatland_random_sparse_small/global_obs_conv_net/ppo.yaml`

Test example:

`python ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run PPO --no-render
        --config '{"env_config": {"test": true}}' --episodes 1000 --out rollouts.pkl`

Note that -f overrides all other trial-specific command-line options.

## Experiment structure

Experiments consist of one or many RLlib YAML config files alongside a MARKDOWN file containing results, plots and a detailed description of the methodology.

All files are stored in a experiment folder under `experiments/<env-name>/<experiment-name>`.

- [Tree observations w/ fully connected network](master/experiments/flatland_random_sparse_small/tree_obs_fc_net)
- [Global observations w/ convnet](Global observation convnet experiments)


Notes
---

- The basic structure of this repository is adopted from [https://github.com/spMohanty/rl-experiments/](https://github.com/spMohanty/rl-experiments/)