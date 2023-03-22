import unittest

import numpy as np

from agents.fm_hdqn_agent import FactoryMachinesHDQNAgent
from factory_machines_env import FactoryMachinesRelativeWrapper
from factory_machines_env.envs import FactoryMachinesEnvMulti


class FMHDQNAgentTest(unittest.TestCase):
    def test_should_report_satisfied_goal(self):
        env = FactoryMachinesRelativeWrapper(FactoryMachinesEnvMulti())
        obs, _ = env.reset()
        agent = FactoryMachinesHDQNAgent(obs, env.action_space.n)

        obs, _, _, _, _ = env.step(env.right)
        obs, _, _, _, _ = env.step(env.right)
        next_obs, _, _, _, _ = env.step(env.grab)

        self.assertTrue(agent.goal_satisfied(obs, env.grab, next_obs, 1))
        self.assertFalse(agent.goal_satisfied(obs, env.grab, next_obs, 2))


if __name__ == '__main__':
    unittest.main()
