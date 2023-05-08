import numpy as np

from factory_machines.agents.h_dqn_agent import HDQNAgent, DictObsType, ActType, FlatObsType


class GridWorldHDQNAgent(HDQNAgent):
    """
    HDQN agent for the GridWorld environment. Just for testing.
    """

    up, left, down, right, grab = range(5)

    def __init__(self, obs: DictObsType, n_actions: int, device: str = 'cpu') -> None:
        n_checkpoints = (len(obs["checkpoints"]) // 2)
        super().__init__(obs, n_checkpoints, n_actions, device=device)

    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        reward = 0
        if self.goal_satisfied(obs, action, next_obs, goal):
            reward += 5
        elif action == self.grab:
            reward += -1

        if self._did_collide(obs["agent_obs"].reshape((3, 3)), action):
            reward += -1

        return reward + -0.01

    def goal_satisfied(self, obs: DictObsType, action: ActType, next_obs, goal: ActType) -> bool:
        offset = goal * 2
        goal_depot_loc = obs["checkpoints"][offset:offset + 2]
        return np.array_equal(goal_depot_loc, [0, 0]) and action == self.grab

    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        return super().to_q1(flatten(obs), goal)  # TODO try with narrower state.

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return flatten(obs)  # TODO try with narrower state.

    def _did_collide(self, local_obs, action) -> bool:
        return (action == self.up and local_obs[0, 1] == 1) \
            or (action == self.down and local_obs[2, 1] == 1) \
            or (action == self.right and local_obs[1, 2] == 1) \
            or (action == self.left and local_obs[1, 0] == 1)


def flatten(obs: DictObsType) -> FlatObsType:
    return [
        *obs["agent_loc"],
        *obs["agent_obs"],
        *obs["checkpoints"],
        *obs["goal"]
    ]
