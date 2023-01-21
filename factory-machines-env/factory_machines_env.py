from typing import Optional, Union, List, Tuple

import gym
from gym import spaces
import pygame
from gym.core import RenderFrame, ActType, ObsType


class FactoryMachinesEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5) -> None:
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "depot": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "steel": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "wood": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass