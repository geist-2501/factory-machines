from typing import Optional, Tuple

import numpy as np
from gym.core import ActType, ObsType
from factory_machines_env.envs.fm_env_base import FactoryMachinesEnvBase


class FactoryMachinesEnvSingle(FactoryMachinesEnvBase):

    def __init__(self, render_mode: Optional[str] = None, map_id="0", order_override: str = None) -> None:
        super().__init__(render_mode, map_id)

        self.order_override = order_override

        self._depot_queues = self._create_order()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, _, _, info = super().step(action)

        terminated = sum(self._depot_queues) == 0
        reward += 50 if terminated else 0

        return obs, reward, terminated, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        obs, _ = super().reset(seed=seed, options=options)

        self._depot_queues = self._create_order()

        return obs, {}

    def _create_order(self):
        if self.order_override:
            order = np.fromstring(self.order_override, sep=',', dtype=int)
        else:
            # Produce a random order.
            order = np.zeros(self._num_depots)
            while sum(order) == 0:
                order = (np.random.normal(size=self._num_depots) > 0.5).astype(int)

        return order

    def _get_depot_queues(self):
        return self._depot_queues

    def _depot_drop_off(self):
        agent_inv_inverse = 1 - self._agent_inv
        drop_off_reward = sum(self._depot_queues * self._agent_inv) * 10
        self._depot_queues *= agent_inv_inverse  # Clear the queues of items the agent had.
        self._agent_inv = np.zeros(self._num_depots, dtype=int)
        self._history.log("Agent dropped off resources.")
        return drop_off_reward

