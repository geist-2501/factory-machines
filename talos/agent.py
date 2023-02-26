from abc import ABC, abstractmethod
from typing import Dict


class Agent(ABC):
    """Base class for an agent for use in the Talos ecosystem."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def get_action(self, state):
        """Request an action."""
        pass

    @abstractmethod
    def save(self) -> Dict:
        """Extract all policy data."""
        pass

    @abstractmethod
    def load(self, agent_data: Dict):
        """Load policy data."""
        pass