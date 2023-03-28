import unittest

from agents.h_dqn_agent import KCatchUpTimeKeeper


class HDQNAgentTest(unittest.TestCase):
    def test_timekeeper_should_allow_q1_training(self):
        timekeeper = KCatchUpTimeKeeper()
        self._step(timekeeper, 3)
        timekeeper.set_k_catch_up(5)
        self.assertTrue(timekeeper.should_train_q1())
        self._step(timekeeper, 3)
        self.assertTrue(timekeeper.should_train_q1())
        self._step(timekeeper, 2)
        self.assertFalse(timekeeper.should_train_q1())
        self._meta(timekeeper, 8)
        self.assertTrue(timekeeper.should_train_q1())

    @staticmethod
    def _step(timekeeper: KCatchUpTimeKeeper, num: int):
        for _ in range(num):
            timekeeper.step()

    @staticmethod
    def _meta(timekeeper: KCatchUpTimeKeeper, num: int):
        for _ in range(num):
            timekeeper.meta_step()

if __name__ == '__main__':
    unittest.main()