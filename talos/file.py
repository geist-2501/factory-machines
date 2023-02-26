from typing import List, Any
from dataclasses import dataclass

import torch
from talos.error import TalfileLoadError


@dataclass()
class TalFile:
    id: str
    iters_trained: int
    record_freq: int
    recorded_rewards: List[float]
    agent_data: Any

    def write(self, path: str):
        with open(path, 'wb') as file:
            torch.save({
                "id": self.id,
                "iters_trained": self.iters_trained,
                "record_freq": self.record_freq,
                "recorded_rewards": self.recorded_rewards,
                "agent_data": self.agent_data
            }, file)


def read_talfile(path: str) -> TalFile:
    try:
        with open(path, 'rb') as file:
            data = torch.load(file)
            return TalFile(**data)
    except OSError as ex:
        raise TalfileLoadError(ex)


if __name__ == '__main__':
    to_write_talfile = TalFile("DQN", 1234, 20, [1, 2, 3, 4], torch.zeros((2, 2)))
    to_write_talfile.write("test.tal")

    print(read_talfile("test.tal"))

