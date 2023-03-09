import numpy as np

from factory_machines.h_dqn_agent import HDQNAgent, DictObsType, FlatObsType, ActType
from factory_machines.utils import flatten


class FactoryMachinesHDQNAgent(HDQNAgent):
    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        return 5 if self.goal_satisfied(obs, goal) else 0

    def goal_satisfied(self, obs: DictObsType, goal: ActType) -> bool:
        target_depot_loc = obs["depot_locs"][goal]  # Assumes relative locations
        return np.array_equal(target_depot_loc, [0, 0])

    def to_q1(self, obs: DictObsType) -> FlatObsType:
        return flatten(obs)  # TODO try with narrower state.

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return flatten(obs)  # TODO try with narrower state.
