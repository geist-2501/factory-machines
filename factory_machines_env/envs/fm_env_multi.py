from typing import Optional, Tuple

import numpy as np
from gym.core import ActType, ObsType
from numpy.random import default_rng
from factory_machines_env.envs.fm_env_base import FactoryMachinesEnvBase


class FactoryMachinesEnvMulti(FactoryMachinesEnvBase):

    def __init__(self, render_mode: Optional[str] = None, map_id=1, num_orders=1) -> None:
        super().__init__(render_mode, map_id)

        self._total_num_orders = num_orders
        self._current_num_orders = num_orders

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        obs, _ = super().reset(seed=seed, options=options)

        self._current_num_orders = self._total_num_orders

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, _, info = super().step(action)

        # Process orders.
        should_create_order = bool(np.random.binomial(1, 0.1))
        if should_create_order:
            order = np.zeros(self._num_depots)
            while sum(order) == 0:
                order = (np.random.normal(size=self._num_depots) > 0.5).astype(int)
            self._depot_queues += order

        terminated = self._current_num_orders == 0 and sum(self._depot_queues) == 0
        reward += 100 if terminated else 0

        return obs, reward, terminated, False, info
