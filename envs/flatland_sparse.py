import logging
from pprint import pprint

import gym
from flatland.envs.malfunction_generators import malfunction_from_params, no_malfunction_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from ray.rllib import MultiAgentEnv

from envs.flatland import get_generator_config
from envs.flatland.observations import make_obs
from envs.flatland.utils.gym_env import FlatlandGymEnv
from envs.flatland.utils.gym_env_wrappers import AvailableActionsWrapper, SkipNoChoiceCellsWrapper


class FlatlandSparse(MultiAgentEnv):
    def __init__(self, env_config) -> None:
        super().__init__()

        # TODO implement other generators
        assert env_config['generator'] == 'sparse_rail_generator'

        self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))
        self._config = get_generator_config(env_config['generator_config'])

        if env_config.worker_index == 0 and env_config.vector_index == 0:
            print("=" * 50)
            pprint(self._config)
            print("=" * 50)

        self._env = FlatlandGymEnv(
            rail_env=self._launch(),
            observation_space=self._observation.observation_space(),
            # render=env_config['render'], # TODO need to fix gl compatibility first
            regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
            regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset']
        )
        if env_config.get('skip_no_choice_cells', False):
            self._env = SkipNoChoiceCellsWrapper(self._env)
        if env_config.get('available_actions_obs', False):
            self._env = AvailableActionsWrapper(self._env)

    @property
    def observation_space(self) -> gym.spaces.Space:
        print(self._env.observation_space)
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    def _launch(self):
        rail_generator = sparse_rail_generator(
            seed=self._config['seed'],
            max_num_cities=self._config['max_num_cities'],
            grid_mode=self._config['grid_mode'],
            max_rails_between_cities=self._config['max_rails_between_cities'],
            max_rails_in_city=self._config['max_rails_in_city']
        )

        malfunction_generator = no_malfunction_generator()
        if {'malfunction_rate', 'min_duration', 'max_duration'} <= self._config.keys():
            stochastic_data = {
                'malfunction_rate': self._config['malfunction_rate'],
                'min_duration': self._config['malfunction_min_duration'],
                'max_duration': self._config['malfunction_max_duration']
            }
            malfunction_generator = malfunction_from_params(stochastic_data)

        speed_ratio_map = None
        if 'speed_ratio_map' in self._config:
            speed_ratio_map = {
                float(k): float(v) for k, v in self._config['speed_ratio_map'].items()
            }
        schedule_generator = sparse_schedule_generator(speed_ratio_map)

        env = None
        try:
            env = RailEnv(
                width=self._config['width'],
                height=self._config['height'],
                rail_generator=rail_generator,
                schedule_generator=schedule_generator,
                number_of_agents=self._config['number_of_agents'],
                malfunction_generator_and_process_data=malfunction_generator,
                obs_builder_object=self._observation.builder(),
                remove_agents_at_target=False,
                random_seed=self._config['seed']
            )

            env.reset()
        except ValueError as e:
            logging.error("=" * 50)
            logging.error(f"Error while creating env: {e}")
            logging.error("=" * 50)

        return env

    def step(self, action_dict):
        return self._env.step(action_dict)

    def reset(self):
        return self._env.reset()
