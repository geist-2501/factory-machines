from dataclasses import dataclass
from typing import Dict, Callable
from talos.agent import Agent
from talos.error import AgentNotFound


@dataclass
class AgentSpec:
    """A specification for creating agents with Talos."""
    id: str
    training_wrapper: Callable
    agent_factory: Callable[[int, int], Agent]


# Global registry for storing agent configs.
registry: Dict[str, AgentSpec] = {}


def register(
        id: str,
        training_wrapper: Callable,
        agent_factory: Callable[[], Agent],
):
    """Register an agent with Talos."""
    global registry

    agent_spec = AgentSpec(
        id=id,
        training_wrapper=training_wrapper,
        agent_factory=agent_factory
    )

    registry[id] = agent_spec


def get_agent(
        id: str
) -> tuple[Callable, Callable]:
    global registry

    if id in registry:
        spec = registry[id]
        return spec.agent_factory, spec.training_wrapper
    else:
        raise AgentNotFound
