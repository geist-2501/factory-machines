import unittest

import numpy as np

from factory_machines.envs import OrderWorldBasic
from factory_machines.envs.order_generators import MockOrderGenerator


class OrderWorldBasicTest(unittest.TestCase):
    def test_should_punish_incorrect_move(self):
        env = OrderWorldBasic(generator=MockOrderGenerator([
            np.array([1, 0, 0])
        ]))

        env.reset()
        _, r1, _, _, _ = env.step(0)
        _, r2, _, _, _ = env.step(1)

        self.assertAlmostEqual(r1, -1 - 2)
        self.assertAlmostEqual(r2, -2 - 2)

    def test_should_give_no_reward_on_output(self):
        env = OrderWorldBasic(generator=MockOrderGenerator([
            None
        ]))

        env.reset()
        _, r, _, _, _ = env.step(3)

        self.assertAlmostEqual(r, -0.5)

    def test_should_give_reward_on_order_complete(self):
        env = OrderWorldBasic(generator=MockOrderGenerator([
            np.array([1, 0, 1])
        ]))

        env.reset()

        env.step(3)  # Must wait one step for an order to come in.

        _, r1, _, _, _ = env.step(0)
        _, r2, _, _, _ = env.step(2)
        _, r3, _, _, _ = env.step(3)

        self.assertAlmostEqual(r1, 1 - 1)
        self.assertAlmostEqual(r2, 1 - 1)
        self.assertAlmostEqual(r3, 12 - 2)


if __name__ == '__main__':
    unittest.main()
