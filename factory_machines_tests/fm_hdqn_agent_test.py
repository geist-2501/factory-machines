import unittest

import numpy as np

from factory_machines.fm_hdqn_agent import FactoryMachinesHDQNAgent
from factory_machines_env import FactoryMachinesRelativeWrapper
from factory_machines_env_tests import MockEnv


class FMHDQNAgentTest(unittest.TestCase):
    def test_should_report_satisfied_goal(self):
        env = FactoryMachinesRelativeWrapper(MockEnv(
            agent_loc=np.array([1, 1]),
            depot_locs=np.array([[1, 1], [2, 3]])
        ))
        obs, _ = env.reset()
        agent = FactoryMachinesHDQNAgent(obs, env.action_space.n)

        self.assertTrue(agent.goal_satisfied(obs, 0))
        self.assertFalse(agent.goal_satisfied(obs, 1))


if __name__ == '__main__':
    unittest.main()
