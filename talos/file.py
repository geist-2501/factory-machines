import io
import json
from typing import List
from dataclasses import dataclass

import torch


@dataclass()
class TalFile:
    id: str
    iters_trained: int
    record_freq: int
    recorded_rewards: List[float]

    def write(self, path: str):
        with open(path, 'wb') as file:
            test = torch.zeros((2, 2))
            torch.save({
                "id": self.id,
                "iters_trained": self.iters_trained,
                "record_freq": self.record_freq,
                "recorded_rewards": self.recorded_rewards,
                "agent_data": test
            }, file)


def read_talfile(path: str) -> TalFile:
    with open(path, 'rb') as file:
        test = torch.load(file)
        print(test)


if __name__ == '__main__':
    to_write_talfile = TalFile("DQN", 1234, 20, [1, 2, 3, 4])
    to_write_talfile.write("test.tal")

    read_talfile("test.tal")

