import unittest

import gym
import numpy as np

from factory_machines.envs import FactoryMachinesEnvMulti, FactoryMachinesEnvMultiV2
from factory_machines.envs.order_generators import MockOrderGenerator


class FMEnvMultiV2Test(unittest.TestCase):

    _up = 0
    _left = 1
    _down = 2
    _right = 3
    _grab = 4

    def test_ages(self):
        order_schedule = [None] * 30
        order_schedule[0] = np.array([1, 0, 0])
        order_schedule[25] = np.array([0, 1, 0])
        env = FactoryMachinesEnvMultiV2(
            map_id="0",
            num_orders=2,
            order_generator=MockOrderGenerator(order_schedule)
        )
        # Map looks like;
        # 'o.1'
        # '...'
        # '2.3'

        env.reset()

        # Bang head against wall until all orders are out.
        self._go(env, self._up, 50)

        ages, _, _ = env.get_info()

        self.assertAlmostEqual(ages[0], 1)
        self.assertAlmostEqual(ages[1], 0.5)

    def test_reward_scheme_1(self):
        order_schedule = [None] * 30
        order_schedule[0] = np.array([1, 0, 0])
        order_schedule[25] = np.array([0, 1, 0])
        env = FactoryMachinesEnvMultiV2(
            map_id="0",
            num_orders=2,
            order_generator=MockOrderGenerator(order_schedule)
        )
        # Map looks like;
        # 'o.1'
        # '...'
        # '2.3'

        env.reset()

        # Bang head against wall until all orders are out.
        self._go(env, self._right, 2)
        env.step(self._grab)
        self._go(env, self._left, 1)
        self._go(env, self._up, 50 - 4)

        s, r, _, _, _ = env.step(self._left)

        self.assertAlmostEqual(r, 10 + 0 - 0.5)

    def test_reward_scheme_2(self):
        order_schedule = [None] * 30
        order_schedule[0] = np.array([1, 0, 0])
        order_schedule[25] = np.array([0, 1, 0])
        env = FactoryMachinesEnvMultiV2(
            map_id="0",
            num_orders=2,
            order_generator=MockOrderGenerator(order_schedule)
        )
        # Map looks like;
        # 'o.1'
        # '...'
        # '2.3'

        env.reset()

        # Bang head against wall until all orders are out.
        self._go(env, self._up, 50 - 5)
        self._go(env, self._down, 2)
        env.step(self._grab)
        self._go(env, self._up, 1)

        s, r, _, _, _ = env.step(self._up)

        self.assertAlmostEqual(r, 10 + 1.5 - 0.5)

    @staticmethod
    def _go(env: gym.Env, direction: int, steps: int):
        for _ in range(steps):
            env.step(direction)


if __name__ == '__main__':
    unittest.main()
