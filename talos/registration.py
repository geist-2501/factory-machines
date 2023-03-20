from dataclasses import dataclass
from typing import Dict, Callable, Tuple

import gym

from talos.agent import Agent
from talos.error import AgentNotFound, WrapperNotFound


@dataclass
class AgentSpec:
    """A specification for creating agents with Talos."""
    id: str
    training_wrapper: Callable
    graphing_wrapper: Callable[[Dict], None]
    agent_factory: Callable[[int, int], Agent]


@dataclass
class WrapperSpec:
    """A specification for using environment wrappers with Talos."""
    id: str
    wrapper_factory: Callable[[gym.Env], gym.Wrapper]


# Global registries for storing configs.
agent_registry: Dict[str, AgentSpec] = {}
wrapper_registry: Dict[str, WrapperSpec] = {}


def _dummy_training_wrapper(
        env_factory,
        agent,
        agent_config,
        training_artifacts
):
    print("Dummy training wrapper in use - no training will be done.")
    pass


def _dummy_graphing_wrapper(
        artifacts
):
    print("Dummy graphing wrapper in use - no graphs produced.")
    pass


def register_agent(
        agent_id: str,
        agent_factory: Callable[[], Agent],
        training_wrapper: Callable = _dummy_training_wrapper,
        graphing_wrapper: Callable[[Dict], None] = _dummy_graphing_wrapper,
):
    """Register an agent with Talos."""
    global agent_registry

    agent_spec = AgentSpec(
        id=agent_id,
        training_wrapper=training_wrapper,
        graphing_wrapper=graphing_wrapper,
        agent_factory=agent_factory
    )

    agent_registry[agent_id] = agent_spec


def _get_spec(agent_id: str) -> AgentSpec:
    global agent_registry

    if agent_id in agent_registry:
        spec = agent_registry[agent_id]
        return spec
    else:
        raise AgentNotFound


def get_agent(
        agent_id: str
) -> Tuple[Callable, Callable]:
    spec = _get_spec(agent_id)
    return spec.agent_factory, spec.training_wrapper


def get_agent_graphing(agent_id: str) -> Callable[[Dict], None]:
    spec = _get_spec(agent_id)
    return spec.graphing_wrapper


def register_wrapper(
        wrapper_id: str,
        wrapper_factory: Callable[[gym.Env], gym.Wrapper]
):
    """Register a wrapper with Talos."""
    global wrapper_registry

    wrapper_spec = WrapperSpec(
        id=wrapper_id,
        wrapper_factory=wrapper_factory
    )

    wrapper_registry[wrapper_id] = wrapper_spec


def get_wrapper(
        id: str
) -> Callable:
    global wrapper_registry

    if id in wrapper_registry:
        spec = wrapper_registry[id]
        return spec.wrapper_factory
    else:
        raise WrapperNotFound
