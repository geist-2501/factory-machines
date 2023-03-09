from typing import Optional, Union, List, Tuple

import gym
import numpy as np
from gym.core import RenderFrame, ActType, ObsType


class MockEnv(gym.Env):
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs = self._make_dummy_obs()

        return obs, 0, False, False, {}

    @staticmethod
    def _make_dummy_obs():
        return {
            "agent_loc": np.array([1, 2]),
            "agent_obs": np.array([[0, 0], [1, 0]]).flatten(),
            "agent_inv": np.array([0]),
            "depot_locs": np.array([[1, 2]]).flatten(),
            "depot_queues": np.array([1]),
            "output_loc": np.array([2, 2]),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        obs = self._make_dummy_obs()

        return obs, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        raise NotImplementedError
