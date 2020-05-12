import multiprocessing
import numbers
from pprint import pprint

import wandb
from ray import tune

# ray 0.8.1 reorganized ray.tune.util -> ray.tune.utils
try:
    from ray.tune.utils import flatten_dict
except ImportError:
    from ray.tune.util import flatten_dict


class WandbLogger(tune.logger.Logger):
    """Pass WandbLogger to the loggers argument of tune.run

       tune.run("PG", loggers=[WandbLogger], config={
           "monitor": True, "env_config": {
               "wandb": {"project": "my-project-name"}}})
    """

    def _init(self):
        self._config = None
        self.metrics_queue_dict = {}

    def on_result(self, result):
        experiment_tag = result.get('experiment_tag', 'no_experiment_tag')
        experiment_id = result.get('experiment_id', 'no_experiment_id')

        if experiment_tag not in self.metrics_queue_dict:
            print("=" * 50)
            print("Setting up new w&b logger")
            print("Experiment tag:", experiment_tag)
            print("Experiment id:", experiment_id)
            config = result.get("config")
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=wandb_process, args=(queue, config,))
            p.start()
            self.metrics_queue_dict[experiment_tag] = queue
            print("=" * 50)

        queue = self.metrics_queue_dict[experiment_tag]

        tmp = result.copy()
        for k in ["done", "config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]

        metrics = {}
        for key, value in flatten_dict(tmp, delimiter="/").items():
            if not isinstance(value, numbers.Number):
                continue
            metrics[key] = value

        queue.put(metrics)

    def close(self):
        wandb.join()


# each logger has to run in a separate process
def wandb_process(queue, config):
    run = wandb.init(reinit=True, **config.get("env_config", {}).get("wandb", {}))

    if config:
        for k in config.keys():
            if k != "callbacks":
                if wandb.config.get(k) is None:
                    wandb.config[k] = config[k]

        if 'yaml_config' in config['env_config']:
            yaml_config = config['env_config']['yaml_config']
            print("Saving full experiment config:", yaml_config)
            wandb.save(yaml_config)

    while True:
        metrics = queue.get()
        run.log(metrics)
