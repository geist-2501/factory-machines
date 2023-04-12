import unittest

import numpy as np

from factory_machines.wrappers import FactoryMachinesFlattenRelativeWrapper
from factory_machines_test.envs import MockEnv


class FMFlattenRelativeTest(unittest.TestCase):
    def test_should_flatten_relative(self):
        wrapper = FactoryMachinesFlattenRelativeWrapper(MockEnv(
            agent_loc=np.array([1, 1]),
            depot_locs=np.array([[1, 1], [2, 3]])
        ))
        wrapper.reset()

        obs, _, _, _, _ = wrapper.step(0)

        self.assertListEqual(obs, [1, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1, 2, 1, 2, 1, 1])


if __name__ == '__main__':
    unittest.main()
