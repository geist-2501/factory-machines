from typing import Any, Dict
from dataclasses import dataclass, field

import torch
from talos.error import TalfileLoadError


@dataclass()
class TalFile:
    id: str
    agent_data: Any

    training_artifacts: Dict
    config: Dict
    used_wrappers: str = None
    env_name: str = None
    env_args: Dict = field(default_factory=dict)

    def write(self, path: str):
        with open(path, 'wb') as file:
            torch.save({
                "id": self.id,
                "agent_data": self.agent_data,
                "training_artifacts": self.training_artifacts,
                "config": self.config,
                "used_wrappers": self.used_wrappers,
                "env_name": self.env_name,
                "env_args": self.env_args
            }, file)


def read_talfile(path: str) -> TalFile:
    try:
        with open(path, 'rb') as file:
            data = torch.load(file)
            return TalFile(**data)
    except OSError as ex:
        raise TalfileLoadError(ex)
