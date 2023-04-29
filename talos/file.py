from typing import Any, Dict, List
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

    def get_artifact(self, path: List[str]) -> Any:
        root = self.training_artifacts
        for part in path:
            if type(root) is dict:
                root = root[part]
            elif type(root) is tuple:
                root = root[int(part)]
            else:
                raise RuntimeError("No more appropriate entries to index!")

        return root

    def set_artifact(self, path: List[str], data: Any):
        root = self.training_artifacts
        for part_idx, part in enumerate(path):
            is_last = part_idx == len(path) - 1
            if type(root) is dict:
                indexer = part
            elif type(root) is tuple:
                indexer = int(part)
            else:
                raise RuntimeError("No more appropriate entries to index!")

            if is_last:
                if type(root) is tuple:
                    mutable_root = list(root)
                    mutable_root[indexer] = data
                    immutable_root = tuple(mutable_root)
                else:
                    root[indexer] = data
            else:
                root = root[indexer]


def read_talfile(path: str) -> TalFile:
    try:
        with open(path, 'rb') as file:
            data = torch.load(file)
            return TalFile(**data)
    except OSError as ex:
        raise TalfileLoadError(ex)
