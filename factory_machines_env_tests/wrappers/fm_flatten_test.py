import unittest

from factory_machines_env.wrappers.fm_flatten import FactoryMachinesFlattenWrapper
from factory_machines_env_tests import MockEnv


class MyTestCase(unittest.TestCase):
    def test_should_flatten_obs(self):
        wrapper = FactoryMachinesFlattenWrapper(MockEnv())
        wrapper.reset()

        obs, _, _, _, _ = wrapper.step(0)

        self.assertListEqual(obs, [1, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 2])


if __name__ == '__main__':
    unittest.main()
