import numpy as np

from agents.h_dqn_agent import HDQNAgent, DictObsType, FlatObsType, ActType


class DiscreteStochasticHDQNAgent(HDQNAgent):

    def __init__(self, obs: DictObsType, n_actions: int, device: str = 'cpu') -> None:
        n_goals = len(obs)
        super().__init__(obs, n_goals, n_actions, device=device)

    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        return 1 if self.goal_satisfied(next_obs, goal) else 0

    def goal_satisfied(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> bool:
        current_state = np.argmax(next_obs)
        return current_state == goal

    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        return super().to_q1(obs, goal)

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return super().to_q2(obs)
