import numpy as np

from agents.h_dqn_agent import HDQNAgent, DictObsType, FlatObsType, ActType
from agents.utils import flatten


class FactoryMachinesHDQNAgent(HDQNAgent):

    _act_grab = 4

    def __init__(self, obs: DictObsType, n_actions: int, device: str = 'cpu') -> None:
        # Depot locations is a flattened x,y list, so half it's size for the total number of locations,
        # and add one for the output location.
        n_goals = (len(obs["depot_locs"]) // 2) + 1
        super().__init__(obs, n_goals, n_actions, device=device)

    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        return 5 if self.goal_satisfied(obs, action, next_obs, goal) else 0

    def goal_satisfied(self, obs: DictObsType, action: ActType, next_obs, goal: ActType) -> bool:
        if goal == self.n_goals - 1:
            # Last goal is the output depot.
            return np.array_equal(obs["output_loc"], [0, 0])
        else:
            offset = goal * 2
            goal_depot_loc = obs["depot_locs"][offset:offset + 2]
            return np.array_equal(goal_depot_loc, [0, 0]) and action == self._act_grab

    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        return super().to_q1(flatten(obs), goal)  # TODO try with narrower state.

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return flatten(obs)  # TODO try with narrower state.
