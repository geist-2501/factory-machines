import unittest

from factory_machines.agents import FactoryMachinesMaskedHDQNAgent
from factory_machines.envs import FactoryMachinesEnvMulti
from factory_machines.wrappers import FactoryMachinesRelativeWrapper


class FMHDQNMaskedAgentTest(unittest.TestCase):
    def test_should_obs_mask_correctly(self):
        env = FactoryMachinesRelativeWrapper(FactoryMachinesEnvMulti())
        obs, _ = env.reset()
        agent = FactoryMachinesMaskedHDQNAgent(obs, env.action_space.n)

        new_q1_obs_0 = agent.to_q1(obs, 0)
        new_q1_obs_1 = agent.to_q1(obs, 1)
        new_q1_obs_2 = agent.to_q1(obs, 2)
        new_q1_obs_3 = agent.to_q1(obs, 3)

        self.assertListEqual(new_q1_obs_0, [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0])
        self.assertListEqual(new_q1_obs_1, [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0])
        self.assertListEqual(new_q1_obs_2, [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 0, 1, 0])
        self.assertListEqual(new_q1_obs_3, [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])