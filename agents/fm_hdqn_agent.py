import numpy as np

from agents.h_dqn_agent import HDQNAgent, DictObsType, FlatObsType, ActType
from agents.utils import flatten


class FactoryMachinesHDQNAgent(HDQNAgent):

    def __init__(self, obs: DictObsType, n_actions: int, device: str = 'cpu') -> None:
        # Depot locations is a flattened x,y list, so half it's size for the total number of locations,
        # and add one for the output location.
        n_goals = (len(obs["depot_locs"]) // 2) + 1
        super().__init__(obs, n_goals, n_actions, device=device)

    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        return 5 if self.goal_satisfied(obs, goal) else 0

    def goal_satisfied(self, obs: DictObsType, goal: ActType) -> bool:
        # TODO add requirement that the material is actually picked up!
        # Can check state + action?
        # TODO incorporate the output depot in this check too!
        offset = goal * 2
        goal_depot_loc = obs["depot_locs"][offset:offset + 2]
        return np.array_equal(goal_depot_loc, [0, 0])

    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        return super().to_q1(flatten(obs), goal)  # TODO try with narrower state.

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return flatten(obs)  # TODO try with narrower state.
