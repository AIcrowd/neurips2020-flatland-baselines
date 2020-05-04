import logging

import gym
import numpy as np
from flatland.envs.malfunction_generators import no_malfunction_generator, malfunction_from_params
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from envs.flatland import get_generator_config
from envs.flatland.observations import make_obs
from envs.flatland.utils.rllib_wrapper import FlatlandRllibWrapper, StepOutput


class FlatlandSingle(gym.Env):
    def render(self, mode='human'):
        pass

    def __init__(self, env_config):
        self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))
        self._config = get_generator_config(env_config['generator_config'])

        self._env = FlatlandRllibWrapper(
            rail_env=self._launch(),
            regenerate_rail_on_reset=self._config['regenerate_rail_on_reset'],
            regenerate_schedule_on_reset=self._config['regenerate_schedule_on_reset']
        )

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

    def step(self, action_list):
        # print("="*50)
        # print(action_dict)

        action_dict = {}
        for i, action in enumerate(action_list):
            action_dict[i] = action

        step_r = self._env.step(action_dict)
        # print(step_r)
        # print("="*50)

        return StepOutput(
            obs=[step for step in step_r.obs.values()],
            reward=np.sum([r for r in step_r.reward.values()]),
            done=all(step_r.done.values()),
            info=step_r.info[0]
        )
        #return step_r

    def reset(self):
        foo = self._env.reset()

        # print("="*50)
        # print(foo)
        # print("="*50)

        return [step for step in foo.values()]
        #return foo

    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = self._observation.observation_space()

        if isinstance(observation_space, gym.spaces.Box):
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._config['number_of_agents'], *observation_space.shape,))
        elif isinstance(observation_space, gym.spaces.Tuple):
            spaces = observation_space.spaces * self._config['number_of_agents']
            return gym.spaces.Tuple(spaces)
        else:
            raise ValueError("Unhandled space:", observation_space.__class__)

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([5] * self._config['number_of_agents'])
