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

        self.assertTrue(agent.goal_satisfied(obs, env.grab, next_obs, 0))
        self.assertFalse(agent.goal_satisfied(obs, env.grab, next_obs, 1))

    def test_should_report_satisfied_goal_on_output_depot(self):
        env = FactoryMachinesRelativeWrapper(FactoryMachinesEnvMulti())
        obs, _ = env.reset()
        agent = FactoryMachinesHDQNAgent(obs, env.action_space.n)

        obs, _, _, _, _ = env.step(env.right)
        next_obs, _, _, _, _ = env.step(env.left)

        self.assertTrue(agent.goal_satisfied(obs, env.right, next_obs, 3))
        self.assertFalse(agent.goal_satisfied(obs, env.left, next_obs, 1))

    def test_should_punish_collide(self):
        env = FactoryMachinesRelativeWrapper(FactoryMachinesEnvMulti())
        obs, _ = env.reset()
        agent = FactoryMachinesHDQNAgent(obs, env.action_space.n)

        obs, _, _, _, _ = env.step(env.right)

        self.assertEqual(agent.get_intrinsic_reward(obs, env.up, obs, 0), -1)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.down, obs, 0), -0.01)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.left, obs, 0), -0.01)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.right, obs, 0), -0.01)

        obs, _, _, _, _ = env.step(env.right)

        self.assertEqual(agent.get_intrinsic_reward(obs, env.up, obs, 0), -1)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.down, obs, 0), -0.01)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.left, obs, 0), -0.01)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.right, obs, 0), -1)

        obs, _, _, _, _ = env.step(env.down)
        obs, _, _, _, _ = env.step(env.down)

        self.assertEqual(agent.get_intrinsic_reward(obs, env.up, obs, 0), -0.01)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.down, obs, 0), -1)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.left, obs, 0), -0.01)
        self.assertEqual(agent.get_intrinsic_reward(obs, env.right, obs, 0), -1)

    def test_should_punish_on_incorrect_grab(self):
        env = FactoryMachinesRelativeWrapper(FactoryMachinesEnvMulti())
        obs, _ = env.reset()
        agent = FactoryMachinesHDQNAgent(obs, env.action_space.n)

        next_obs, _, _, _, _ = env.step(env.grab)

        self.assertEqual(agent.get_intrinsic_reward(obs, env.grab, next_obs, 1), -1)

    def test_should_onehot_encode_correctly(self):
        env = FactoryMachinesRelativeWrapper(FactoryMachinesEnvMulti())
        obs, _ = env.reset()
        agent = FactoryMachinesHDQNAgent(obs, env.action_space.n)

        onehot = agent.onehot(1, 3)

        self.assertListEqual(onehot, [0, 1, 0])


if __name__ == '__main__':
    unittest.main()
