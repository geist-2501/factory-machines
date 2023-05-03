from typing import Optional, Union, List, Tuple

import gym
import numpy as np
from gym.core import RenderFrame, ActType, ObsType
from gym import spaces

from factory_machines.envs import FactoryMachinesEnvMulti
from factory_machines.envs.order_generators import OrderGenerator


class MockEnv(gym.Env):
    """A mock environment that mimics the observations of the factory machines environments."""

    def __init__(
            self,
            agent_loc=np.array([1, 2]),
            depot_locs=np.array([[1, 1]]),
            output_loc=np.array([2, 2])
    ):
        self._agent_loc = agent_loc
        self._depot_locs = depot_locs
        self._output_loc = output_loc

        self.action_space = spaces.Discrete(6)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs = self._make_dummy_obs()

        return obs, 0, False, False, {}

    def _make_dummy_obs(self):
        dummy_queues = [x + 1 for x in range(len(self._depot_locs))]
        return {
            "agent_loc": self._agent_loc,
            "agent_obs": np.array([[0, 0], [1, 0]]).flatten(),
            "agent_inv": np.array(dummy_queues),
            "depot_locs": self._depot_locs.flatten(),
            "depot_queues": np.array(dummy_queues),
            "depot_ages": np.array(dummy_queues) * 2,
            "output_loc": self._output_loc,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        return self._make_dummy_obs(), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        raise NotImplementedError
