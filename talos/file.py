from typing import List, Any, Dict
from dataclasses import dataclass

import torch
from talos.error import TalfileLoadError


@dataclass()
class TalFile:
    id: str
    agent_data: Any

    training_artifacts: Dict
    used_wrappers: str = None
    env_name: str = None

    def write(self, path: str):
        with open(path, 'wb') as file:
            torch.save({
                "id": self.id,
                "agent_data": self.agent_data,
                "training_artifacts": self.training_artifacts,
                "used_wrappers": self.used_wrappers,
                "env_name": self.env_name
            }, file)


def read_talfile(path: str) -> TalFile:
    try:
        with open(path, 'rb') as file:
            data = torch.load(file)
            return TalFile(**data)
    except OSError as ex:
        raise TalfileLoadError(ex)
