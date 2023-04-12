import unittest

from factory_machines.agents.timekeeper import KCatchUpTimeKeeper, SerialTimekeeper


def repeat(func, iters):
    for _ in range(iters):
        func()


class TimekeeperTest(unittest.TestCase):
    def test_should_get_correct_train_values(self):
        timekeeper = KCatchUpTimeKeeper()

        timekeeper.pretrain_mode()
        self.assertTrue(timekeeper.should_train_q1())
        self.assertFalse(timekeeper.should_train_q2())

        timekeeper.train_mode()
        self.assertTrue(timekeeper.should_train_q1())
        self.assertTrue(timekeeper.should_train_q2())

    def test_should_get_correct_k_catch_up_values(self):
        timekeeper = KCatchUpTimeKeeper()

        timekeeper.train_mode()
        timekeeper.set_k_catch_up(2)
        self.assertTrue(timekeeper.should_train_q1())
        self.assertTrue(timekeeper.should_train_q2())
        timekeeper.step_q1()
        timekeeper.step_q1()
        self.assertTrue(timekeeper.should_train_q1())
        self.assertTrue(timekeeper.should_train_q2())
        timekeeper.step_q1()
        self.assertFalse(timekeeper.should_train_q1())
        self.assertTrue(timekeeper.should_train_q2())

    def test_timekeeper_should_allow_q1_training(self):
        timekeeper = KCatchUpTimeKeeper()
        repeat(lambda: timekeeper.step_q1(), 3)
        timekeeper.set_k_catch_up(5)
        self.assertTrue(timekeeper.should_train_q1())
        self._step(timekeeper, 3)
        self.assertTrue(timekeeper.should_train_q1())
        self._step(timekeeper, 2)
        self.assertFalse(timekeeper.should_train_q1())
        self._meta(timekeeper, 8)
        self.assertTrue(timekeeper.should_train_q1())

    def test_2_serial_timekeeper_should_select_correct_training_windows(self):
        timekeeper = SerialTimekeeper()

        timekeeper.pretrain_mode()
        self.assertTrue(timekeeper.should_train_q1())
        self.assertFalse(timekeeper.should_train_q2())

        timekeeper.train_mode()
        self.assertFalse(timekeeper.should_train_q1())
        self.assertTrue(timekeeper.should_train_q2())

    @staticmethod
    def _step(timekeeper: KCatchUpTimeKeeper, num: int):
        for _ in range(num):
            timekeeper.step_q1()

    @staticmethod
    def _meta(timekeeper: KCatchUpTimeKeeper, num: int):
        for _ in range(num):
            timekeeper.step_q2()

if __name__ == '__main__':
    unittest.main()