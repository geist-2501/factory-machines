import configparser
from typing import Callable, Dict, Optional, Any

from gym import Env

from talos.agent import Agent

AgentFactory = Callable[[Any, int, str], Agent]
EnvFactory = Callable[[int], Env]

Artifacts = Dict
SaveCallback = Callable[[Any, Artifacts, int, Optional[str]], None]