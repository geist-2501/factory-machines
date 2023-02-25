from dataclasses import dataclass
from typing import Dict, Callable

import gym

from talos.agent import Agent
from talos.error import AgentNotFound, WrapperNotFound


@dataclass
class AgentSpec:
    """A specification for creating agents with Talos."""
    id: str
    training_wrapper: Callable
    agent_factory: Callable[[int, int], Agent]


@dataclass
class WrapperSpec:
    """A specification for using environment wrappers with Talos."""
    id: str
    wrapper_factory: Callable[[gym.Env], gym.Wrapper]


# Global registries for storing configs.
agent_registry: Dict[str, AgentSpec] = {}
wrapper_registry: Dict[str, WrapperSpec] = {}


def register_agent(
        id: str,
        training_wrapper: Callable,
        agent_factory: Callable[[], Agent],
):
    """Register an agent with Talos."""
    global agent_registry

    agent_spec = AgentSpec(
        id=id,
        training_wrapper=training_wrapper,
        agent_factory=agent_factory
    )

    agent_registry[id] = agent_spec


def get_agent(
        id: str
) -> tuple[Callable, Callable]:
    global agent_registry

    if id in agent_registry:
        spec = agent_registry[id]
        return spec.agent_factory, spec.training_wrapper
    else:
        raise AgentNotFound


def register_wrapper(
        id: str,
        wrapper_factory: Callable[[gym.Env], gym.Wrapper]
):
    """Register a wrapper with Talos."""
    global wrapper_registry

    wrapper_spec = WrapperSpec(
        id=id,
        wrapper_factory=wrapper_factory
    )

    wrapper_registry[id] = wrapper_spec


def get_wrapper(
        id: str
) -> Callable:
    global wrapper_registry

    if id in wrapper_registry:
        spec = wrapper_registry[id]
        return spec.wrapper_factory
    else:
        raise WrapperNotFound
