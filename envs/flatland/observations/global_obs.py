import gym
import numpy as np
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import GlobalObsForRailEnv

from envs.flatland.observations import Observation, register_obs


@register_obs("global")
class GlobalObservation(Observation):

    def __init__(self, config) -> None:
        super().__init__(config)
        self._config = config
        self._builder = PaddedGlobalObsForRailEnv(max_width=config['max_width'], max_height=config['max_height'])

    def builder(self) -> ObservationBuilder:
        return self._builder

    def observation_space(self) -> gym.Space:
        grid_shape = (self._config['max_width'], self._config['max_height'])
        return gym.spaces.Tuple([
            gym.spaces.Box(low=0, high=np.inf, shape=grid_shape + (16,), dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, shape=grid_shape + (5,), dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, shape=grid_shape + (2,), dtype=np.float32),
        ])


class PaddedGlobalObsForRailEnv(ObservationBuilder):

    def __init__(self, max_width, max_height):
        super().__init__()
        self._max_width = max_width
        self._max_height = max_height
        self._builder = GlobalObsForRailEnv()

    def set_env(self, env: Environment):
        self._builder.set_env(env)

    def reset(self):
        self._builder.reset()

    def get(self, handle: int = 0):
        obs = list(self._builder.get(handle))
        height, width = obs[0].shape[:2]
        pad_height, pad_width = self._max_height - height, self._max_width - width
        obs[1] = obs[1] + 1  # get rid of -1
        assert pad_height >= 0 and pad_width >= 0
        return tuple([
            np.pad(o, ((0, pad_height), (0, pad_height), (0, 0)), constant_values=0)
            for o in obs
        ])
