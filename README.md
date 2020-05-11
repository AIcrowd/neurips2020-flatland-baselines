# NeurIPS 2020 Flatland Challenge baselines

📈 [**Results**](https://app.wandb.ai/masterscrat/flatland/reports/Flatland-Baselines--Vmlldzo4OTc5NA) 

Experiments
---

Experiments consist of one or many RLlib YAML config files alongside a MARKDOWN file containing results, plots and a detailed description of the methodology.

All files are stored in a experiment folder under `experiments/<env-name>/<experiment-name>`.

- [Tree observations w/ fully connected network](experiments/flatland_random_sparse_small/tree_obs_fc_net)
- [Global observations w/ convnet](experiments/flatland_random_sparse_small/global_obs_conv_net)

Setup
---

Using conda (recommended):

```
# with GPU support:
conda env create -f environment-gpu.yml

# or, without GPU support:
#conda env create -f environment-cpu.yml

conda activate flatland-env
pip install -r requirements.txt
```

Using pip:

```
# no GPU support:
pip install -r requirements.txt
```

You may need to install/update bazel: [Ubuntu guide](https://docs.bazel.build/versions/master/install-ubuntu.html)

## Usage

Training example:

`python ./train.py -f experiments/flatland_random_sparse_small/global_obs_conv_net/ppo.yaml`

Evaluation example:

`python ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run PPO --no-render
        --config '{"env_config": {"test": true}}' --episodes 1000 --out rollouts.pkl`

Note that -f overrides all other trial-specific command-line options.

Notes
---

- The basic structure of this repository is adopted from [https://github.com/spMohanty/rl-experiments/](https://github.com/spMohanty/rl-experiments/)