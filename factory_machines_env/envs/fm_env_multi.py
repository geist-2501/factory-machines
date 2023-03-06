from typing import Optional, Tuple

import numpy as np
from gym.core import ActType, ObsType
from numpy.random import default_rng
from factory_machines_env.envs.fm_env_base import FactoryMachinesEnvBase
from factory_machines_env.envs.pygame_utils import draw_lines


class FactoryMachinesEnvMulti(FactoryMachinesEnvBase):

    def __init__(self, render_mode: Optional[str] = None, map_id="0", num_orders=1) -> None:
        super().__init__(render_mode, map_id)

        self._total_num_orders = num_orders
        self._num_orders_pending = num_orders
        self._open_orders = {}
        self._completed_orders = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        obs, _ = super().reset(seed=seed, options=options)

        self._num_orders_pending = self._total_num_orders
        self._open_orders = {}
        self._completed_orders = {}

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, _, info = super().step(action)

        # Process orders.
        should_create_order = bool(np.random.binomial(1, 0.1))
        if should_create_order:
            self._num_orders_pending -= 1
            order_id = self._total_num_orders - self._num_orders_pending + 1
            order = np.zeros(self._num_depots, dtype=int)
            while sum(order) == 0:
                order = (np.random.normal(size=self._num_depots) > 0.5).astype(int)
            self._open_orders[order_id] = order

        terminated = len(self._completed_orders) == self._total_num_orders
        reward += 100 if terminated else 0

        return obs, reward, terminated, False, info

    def _render_info(self, font, header_origin, screen_width, spacing):

        # Draw table header.
        table_rows = [
            "   | " + ' '.join(["D" + str(x+1) for x in range(self._num_depots)]),
            "INV| " + self._add_table_padding(self._agent_inv),
            "DEP| " + self._add_table_padding(self._get_depot_queues())
        ]

        for order_id, order in self._open_orders.items():
            table_rows.append(f"O{order_id:02}| " + self._add_table_padding(order))

        draw_lines(table_rows, self.screen, header_origin, font, self.colors["text"])

    @staticmethod
    def _add_table_padding(arr):
        return ' '.join(map(lambda i: f"{i:2}", arr))

    def _depot_drop_off(self) -> int:
        # Go through each open order and strike off items the agent holds.
        # Then, if an order has been completed, move it to the completed pile.
        reward = 0
        for order_id, order in self._open_orders.copy().items():
            reward += sum(order * self._agent_inv)
            agent_inv_inverse = 1 - self._agent_inv
            new_order = order * agent_inv_inverse  # Clears pending items in an order.
            if sum(new_order) == 0:
                # Order is complete!
                reward += 10
                del self._open_orders[order_id]
            else:
                self._open_orders[order_id] = new_order

        self._agent_inv = np.zeros(self._num_depots, dtype=int)

        return reward

    def _get_depot_queues(self):
        depot_queues = np.zeros(self._num_depots, dtype=int)
        for _, order in self._open_orders.items():
            depot_queues += order

        return depot_queues
