import unittest

import gym
import numpy as np

from factory_machines_env.envs import FactoryMachinesEnvMulti
from factory_machines_env.envs.order_generators import MockOrderGenerator


class FMEnvMultiTest(unittest.TestCase):

    _up = 0
    _left = 1
    _down = 2
    _right = 3
    _grab = 4

    # 0: np.array([0, -1], dtype=int),  # w 0, -1
    # 1: np.array([-1, 0], dtype=int),  # a -1, 0
    # 2: np.array([0, 1], dtype=int),  # s 0, 1
    # 3: np.array([1, 0], dtype=int),  # d 1, 0

    def test_should_terminate(self):
        # Map looks like;
        # 'o.w.d'
        # '..w..'
        # '.....'
        # 'd...d'
        env = FactoryMachinesEnvMulti(
            map_id="1",
            num_orders=2,
            order_generator=MockOrderGenerator([
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
            ])
        )

        self._go(env, self._down, 3)
        _, r, _, _, _ = env.step(self._grab)
        self.assertAlmostEqual(r, 0.4)
        self._go(env, self._right, 4)
        _, r, _, _, _ = env.step(self._grab)
        self.assertAlmostEqual(r, 0.4)
        self._go(env, self._left, 4)
        self._go(env, self._up, 2)
        _, r, term, _, _ = env.step(self._up)

        self.assertAlmostEqual(r, 100 + 20 + 2 + 6 - 0.1)
        self.assertTrue(term)

    def test_should_get_correct_depot_ages(self):
        order_schedule = [None] * 120
        order_schedule[12] = np.array([0, 1, 0])
        order_schedule[56] = np.array([0, 1, 1])
        order_schedule[110] = np.array([1, 0, 0])
        env = FactoryMachinesEnvMulti(
            map_id="1",
            num_orders=3,
            order_generator=MockOrderGenerator(order_schedule)
        )
        # Map looks like;
        # 'o.1'
        # '...'
        # '2.3'

        # Bang head against wall until all orders are out.
        self._go(env, self._up, 120)

        ages, _, _ = env.get_info()

        self.assertEqual(ages[0], 0)  # 10 ts old.
        self.assertEqual(ages[1], 3)  # 108 ts old.
        self.assertEqual(ages[2], 1)  # 64 ts old.

        pass

    @staticmethod
    def _go(env: gym.Env, direction: int, steps: int):
        for _ in range(steps):
            env.step(direction)


if __name__ == '__main__':
    unittest.main()
