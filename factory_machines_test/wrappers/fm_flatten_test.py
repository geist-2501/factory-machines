import unittest

from factory_machines.wrappers import FactoryMachinesFlattenWrapper
from factory_machines_test.envs import MockEnv


class FMFlattenTest(unittest.TestCase):
    def test_should_flatten_obs(self):
        wrapper = FactoryMachinesFlattenWrapper(MockEnv())
        wrapper.reset()

        obs, _, _, _, _ = wrapper.step(0)

        self.assertListEqual(obs, [1, 2, 0, 0, 1, 0, 1, 1, 1, 1, 2, 2])


if __name__ == '__main__':
    unittest.main()
