from typing import Dict, Any, Optional, Set, List

import gym
import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions

from envs.flatland.utils.gym_env import StepOutput


def available_actions(env: RailEnv, agent: EnvAgent, allow_noop=True) -> List[int]:
    if agent.position is None:
        return [1] * len(RailEnvActions)
    else:
        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
    # some actions are always available:
    available_acts = [0] * len(RailEnvActions)
    available_acts[RailEnvActions.MOVE_FORWARD] = 1
    available_acts[RailEnvActions.STOP_MOVING] = 1
    if allow_noop:
        available_acts[RailEnvActions.DO_NOTHING] = 1
    # check if turn left/right are available:
    for movement in range(4):
        if possible_transitions[movement]:
            if movement == (agent.direction + 1) % 4:
                available_acts[RailEnvActions.MOVE_RIGHT] = 1
            elif movement == (agent.direction - 1) % 4:
                available_acts[RailEnvActions.MOVE_LEFT] = 1
    return available_acts


class AvailableActionsWrapper(gym.Wrapper):

    def __init__(self, env, allow_noop=True) -> None:
        super().__init__(env)
        self._allow_noop = allow_noop
        self.observation_space = gym.spaces.Dict({
            'obs': self.env.observation_space,
            'available_actions': gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int32)
        })

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        obs, reward, done, info = self.env.step(action_dict)
        return StepOutput(self._transform_obs(obs), reward, done, info)

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        return self._transform_obs(self.env.reset(random_seed))

    def _transform_obs(self, obs):
        rail_env = self.unwrapped.rail_env
        return {
            agent_id: {
                'obs': agent_obs,
                'available_actions': np.asarray(available_actions(rail_env, rail_env.agents[agent_id], self._allow_noop))
            } for agent_id, agent_obs in obs.items()
        }


def find_all_cells_where_agent_can_choose(rail_env: RailEnv):
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(rail_env.height):
        for w in range(rail_env.width):
            pos = (w, h)
            is_switch = False
            # Check for switch: if there is more than one outgoing transition
            for orientation in directions:
                possible_transitions = rail_env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = rail_env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    decision_cells = switches + switches_neighbors
    return tuple(map(set, (switches, switches_neighbors, decision_cells)))


class SkipNoChoiceCellsWrapper(gym.Wrapper):

    def __init__(self, env) -> None:
        super().__init__(env)
        self._switches = None
        self._switches_neighbors = None
        self._decision_cells = None

    def _on_decision_cell(self, agent: EnvAgent):
        return agent.position is None or agent.position in self._decision_cells

    def _on_switch(self, agent: EnvAgent):
        return agent.position in self._switches

    def _next_to_switch(self, agent: EnvAgent):
        return agent.position in self._switches_neighbors

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
        o, r, d, i = {}, {}, {}, {}
        while len(o) == 0:
            obs, reward, done, info = self.env.step(action_dict)
            for agent_id, agent_obs in obs.items():
                if done[agent_id] or self._on_decision_cell(self.unwrapped.rail_env.agents[agent_id]):
                    o[agent_id] = agent_obs
                    r[agent_id] = reward[agent_id]
                    d[agent_id] = done[agent_id]
                    i[agent_id] = info[agent_id]
            d['__all__'] = done['__all__']
            action_dict = {}
        return StepOutput(o, r, d, i)

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        obs = self.env.reset(random_seed)
        self._switches, self._switches_neighbors, self._decision_cells = \
            find_all_cells_where_agent_can_choose(self.unwrapped.rail_env)
        return obs

