import gym
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from ray.rllib import MultiAgentEnv

from envs.flatland.observations import make_obs
from envs.flatland.rllib_wrapper import FlatlandRllibWrapper


class FlatlandSparse(MultiAgentEnv):
    def __init__(self, env_config) -> None:
        super().__init__()
        self._config = env_config
        self._observation = make_obs(env_config['observation'], env_config.get('observation_config'))
        self._env = FlatlandRllibWrapper(rail_env=self._launch(), render=env_config['render'],
                                         regenerate_rail_on_reset=env_config['regenerate_rail_on_reset'],
                                         regenerate_schedule_on_reset=env_config['regenerate_schedule_on_reset'])

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation.observation_space()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    def _launch(self):
        rail_generator = sparse_rail_generator(seed=self._config['seed'], max_num_cities=self._config['max_num_cities'],
                                               grid_mode=self._config['grid_mode'],
                                               max_rails_between_cities=self._config['max_rails_between_cities'],
                                               max_rails_in_city=self._config['max_rails_in_city'])

        stochastic_data = {'malfunction_rate': self._config['malfunction_rate'],
                           'min_duration': self._config['malfunction_min_duration'],
                           'max_duration': self._config['malfunction_max_duration']}

        schedule_generator = sparse_schedule_generator({float(k): float(v)
                                                        for k, v in self._config['speed_ratio_map'].items()})

        env = RailEnv(
            width=self._config['width'],
            height=self._config['height'],
            rail_generator=rail_generator,
            schedule_generator=schedule_generator,
            number_of_agents=self._config['number_of_agents'],
            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
            obs_builder_object=self._observation.builder(),
            remove_agents_at_target=False
        )
        return env

    def step(self, action_dict):
        return self._env.step(action_dict)

    def reset(self):
        return self._env.reset()
