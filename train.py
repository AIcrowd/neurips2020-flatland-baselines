#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import yaml

import ray
from ray.cluster_utils import Cluster
from ray.rllib.evaluation import MultiAgentEpisode
from ray.tune.config_parser import make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from utils.loader import load_envs, load_models


# Try to import both backends for flag checking/warnings.
tf = try_import_tf()
torch, _ = try_import_torch()

EXAMPLE_USAGE = """
Training example:
    python ./train.py --run DQN --env CartPole-v0 --no-log-flatland-stats

Training with Config:
    python ./train.py -f experiments/flatland_random_sparse_small/global_obs/ppo.yaml


Note that -f overrides all other trial-specific command-line options.
"""

# Register all necessary assets in tune registries
load_envs(os.getcwd())  # Load envs
load_models(os.getcwd())  # Load models


def on_episode_end(info):
    episode = info["episode"]  # type: MultiAgentEpisode

    episode_steps = 0
    episode_max_steps = 0
    episode_num_agents = 0
    episode_score = 0
    episode_done_agents = 0

    for agent, agent_info in episode._agent_to_last_info.items():
        if episode_max_steps == 0:
            episode_max_steps = agent_info["max_episode_steps"]
            episode_num_agents = agent_info["num_agents"]
        episode_steps = max(episode_steps, agent_info["agent_step"])
        episode_score += agent_info["agent_score"]
        if agent_info["agent_done"]:
            episode_done_agents += 1

    assert len(episode._agent_to_last_info) == episode_num_agents

    norm_factor = 1.0 / (episode_max_steps + episode_num_agents)
    percentage_complete = float(episode_done_agents) / episode_num_agents

    episode.custom_metrics["episode_steps"] = episode_steps
    episode.custom_metrics["episode_max_steps"] = episode_max_steps
    episode.custom_metrics["episode_num_agents"] = episode_num_agents
    episode.custom_metrics["episode_return"] = episode.total_reward
    episode.custom_metrics["episode_score"] = episode_score
    episode.custom_metrics["episode_score_normalized"] = episode_score * norm_factor
    episode.custom_metrics["percentage_complete"] = percentage_complete


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.")
    parser.add_argument(
        "--ray-redis-max-memory",
        default=None,
        type=int,
        help="--redis-max-memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-memory",
        default=None,
        type=int,
        help="--memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="Local dir to save training results to. Defaults to '{}'.".format(
            DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="Optional URI to sync training results to (e.g. s3://bucket).")
    parser.add_argument(
        "-v", action="store_true", help="Whether to use INFO level logging.")
    parser.add_argument(
        "-vv", action="store_true", help="Whether to use DEBUG level logging.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Whether to use PyTorch (instead of tf) as the DL framework.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Whether to attempt to enable TF eager execution.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to attempt to enable tracing for eager mode.")
    parser.add_argument(
        "--log-flatland-stats",
        action="store_true",
        default=True,
        help="Whether to log additional flatland specfic metrics such as percentage complete or normalized score.")
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    verbose = 1
    for exp in experiments.values():
        # Bazel makes it hard to find files specified in `args` (and `data`).
        # Look for them here.
        # NOTE: Some of our yaml files don't have a `config` section.
        if exp.get("config", {}).get("input") and \
                not os.path.exists(exp["config"]["input"]):
            # This script runs in the ray/rllib dir.
            rllib_dir = Path(__file__).parent
            input_file = rllib_dir.absolute().joinpath(exp["config"]["input"])
            exp["config"]["input"] = str(input_file)

        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")
        if args.eager:
            exp["config"]["eager"] = True
        if args.torch:
            exp["config"]["use_pytorch"] = True
        if args.v:
            exp["config"]["log_level"] = "INFO"
            verbose = 2
        if args.vv:
            exp["config"]["log_level"] = "DEBUG"
            verbose = 3
        if args.trace:
            if not exp["config"].get("eager"):
                raise ValueError("Must enable --eager to enable tracing.")
            exp["config"]["eager_tracing"] = True
        if args.log_flatland_stats:
            exp['config']['callbacks'] = {
                'on_episode_end': on_episode_end,
            }

    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory)
        ray.init(address=cluster.address)
    else:
        ray.init(
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            memory=args.ray_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus)
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
        verbose=verbose,
        concurrent=True)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
