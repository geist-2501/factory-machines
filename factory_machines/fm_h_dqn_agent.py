import numpy as np

from factory_machines.h_dqn_agent import HDQNAgent


class FactoryMachinesHDQNAgent(HDQNAgent):
    """A h-DQN agent specifically adapted to the Factory Machines environments."""

    def get_intrinsic_reward(self, obs: np.ndarray, action: int, next_obs: np.ndarray, goal: int) -> float:
        return super().get_intrinsic_reward(obs, action, next_obs, goal)

    def goal_satisfied(self, obs: np.ndarray, goal: int) -> bool:
        return super().goal_satisfied(obs, goal)

