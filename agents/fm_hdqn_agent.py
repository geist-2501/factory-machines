import numpy as np

from agents.h_dqn_agent import HDQNAgent, DictObsType, FlatObsType, ActType
from agents.utils import flatten


class FactoryMachinesHDQNAgent(HDQNAgent):

    _up, _left, _right, _down, _grab = range(5)

    def __init__(self, obs: DictObsType, n_actions: int, device: str = 'cpu') -> None:
        # Depot locations is a flattened x,y list, so half it's size for the total number of locations,
        # and add one for the output location.
        n_goals = (len(obs["depot_locs"]) // 2) + 1
        super().__init__(obs, n_goals, n_actions, device=device)

    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        if self.goal_satisfied(obs, action, next_obs, goal):
            return 1

        if action == self._grab:
            return -1

        if self._did_collide(obs["agent_obs"].reshape((3, 3)), action):
            return -1

        return -0.1

    def goal_satisfied(self, obs: DictObsType, action: ActType, next_obs, goal: ActType) -> bool:
        if goal == self.n_goals - 1:
            # Last goal is the output depot.
            equal = np.array_equal(next_obs["output_loc"], [0, 0])
            return equal
        else:
            offset = goal * 2
            goal_depot_loc = obs["depot_locs"][offset:offset + 2]
            return np.array_equal(goal_depot_loc, [0, 0]) and action == self._grab

    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        return super().to_q1(flatten(obs), goal)  # TODO try with narrower state.

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return flatten(obs)  # TODO try with narrower state.

    def _did_collide(self, local_obs, action) -> bool:
        return (action == self._up and local_obs[-1, 0] == 1) \
            or (action == self._down and local_obs[1, 0] == 1) \
            or (action == self._right and local_obs[0, 1] == 1) \
            or (action == self._left and local_obs[0, -1] == 1)
