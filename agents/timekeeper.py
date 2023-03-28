from typing import Optional


class KCatchUpTimeKeeper:
    def __init__(self) -> None:
        super().__init__()
        self._in_pretrain_mode = False
        self._pretrain_steps = 0
        self._steps = 0
        self._meta_steps = 0
        self._episodes = 0
        self.k = None
        self._k_end = 0

    def set_k_catch_up(self, k: Optional[int]):
        if k is None:
            return
        self.k = k
        self._k_end = self._steps + self.k

    def should_train_q1(self) -> bool:
        if self.k is None:
            return True
        else:
            return not self._steps > self._k_end

    def should_train_q2(self) -> bool:
        return not self._in_pretrain_mode

    def step(self):
        if self._in_pretrain_mode:
            self._pretrain_steps += 1
        else:
            self._steps += 1

    def get_steps(self):
        if self._in_pretrain_mode:
            return self._pretrain_steps
        else:
            return self._steps

    def meta_step(self):
        self._meta_steps += 1
        if self.k is not None and self._meta_steps >= self._k_end:
            self._k_end = self._meta_steps + self.k

    def get_meta_steps(self):
        return self._meta_steps

    def episode_step(self):
        self._episodes += 1

    def get_episode_steps(self):
        return self._episodes

    def train_mode(self):
        self._in_pretrain_mode = False

    def pretrain_mode(self):
        self._in_pretrain_mode = True

class SerialTimekeeper(KCatchUpTimeKeeper):
    """Timekeeper class that implements a seperate 2-stage training program."""
