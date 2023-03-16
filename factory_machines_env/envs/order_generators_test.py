import unittest

from factory_machines_env.envs.order_generators import MockOrderGenerator


class OrderGeneratorsTest(unittest.TestCase):
    def test_mock_should_obey_schedule(self):
        num_depots = 2
        gen = MockOrderGenerator([
            [1, 0],
            None,
            None,
            [0, 1],
            None,
            [1, 1],
        ])

        self.assertTrue(gen.should_make_order(0))
        order = gen.make_order(num_depots)
        self.assertListEqual(order, [1, 0])
        self.assertFalse(gen.should_make_order(0))
        self.assertFalse(gen.should_make_order(0))
        self.assertTrue(gen.should_make_order(0))
        order = gen.make_order(num_depots)
        self.assertListEqual(order, [0, 1])
        self.assertFalse(gen.should_make_order(0))
        self.assertTrue(gen.should_make_order(0))
        order = gen.make_order(num_depots)
        self.assertListEqual(order, [1, 1])
        self.assertFalse(gen.should_make_order(0))


if __name__ == '__main__':
    unittest.main()
