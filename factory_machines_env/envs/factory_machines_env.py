from typing import Optional, Union, List, Tuple

import gym
from gym import spaces
import pygame
import numpy as np
from gym.core import RenderFrame, ActType, ObsType


class FactoryMachinesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, size=5, recipe_complexity=3) -> None:
        self.size = size  # The size of the square grid.
        self.window_size = 512  # The size of the PyGame window.
        self.window_header = 100  # The size of the information header.
        self.recipe_complexity = recipe_complexity

        self.res_out = 0
        self.res_steel = 1
        self.res_wood = 2

        self._piles = np.zeros((3, 2), dtype=int)
        self._piles[self.res_out] = [0, 0]
        self._piles[self.res_steel] = [0, size - 1]
        self._piles[self.res_wood] = [size - 1, 0]

        self._pile_colours = [(0, 0, 0)] * 3
        self._pile_colours[self.res_out] = (80, 130, 250)
        self._pile_colours[self.res_wood] = (70, 50, 0)
        self._pile_colours[self.res_steel] = (168, 168, 168)

        self._required = {
            self.res_steel: 0,
            self.res_wood: 0
        }

        self._carrying = 0  # O - nothing, 1 - steel, 2 - wood.

        self._agent = np.array([])

        self.observation_space = spaces.Dict(
            {
                "out_pile": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "steel_pile": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "wood_pile": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "req_wood": spaces.Discrete(recipe_complexity),
                "req_steel": spaces.Discrete(recipe_complexity),
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "carrying": spaces.Discrete(3)
            }
        )

        self.action_space = spaces.Discrete(6)  # Up, down, left, right, grab, drop.

        # Utility vectors for moving the agent.
        self._action_to_direction = {
            0: np.array([0, -1]),  # w 0, -1
            1: np.array([-1, 0]),  # a -1, 0
            2: np.array([0, 1]),  # s 0, 1
            3: np.array([1, 0]),  # d 1, 0
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Used for human friendly rendering.
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._agent,
            "carrying": self._carrying,
            "out_pile": self._piles[self.res_out],
            "steel_pile": self._piles[self.res_steel],
            "wood_pile": self._piles[self.res_wood],
            "req_steel": self._required[self.res_steel],
            "req_wood": self._required[self.res_wood],
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        self._agent = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._required[self.res_wood] = self.np_random.integers(1, self.recipe_complexity, endpoint=True, dtype=int)
        self._required[self.res_steel] = self.np_random.integers(0, self.recipe_complexity, endpoint=True, dtype=int)

        self._carrying = 0

        obs = self._get_obs()

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        illegal_move_punishment = 0
        if action < 4:
            # Action is a move op.
            direction = self._action_to_direction[action]
            new_pos = self._agent + direction
            self._agent = np.clip(new_pos, 0, self.size - 1)
        elif action == 4:
            # Action is a grab op.
            if self._try_grab() is False:
                illegal_move_punishment = 5
        elif action == 5:
            # Action is a drop op.
            if self._try_drop() is False:
                illegal_move_punishment = 5

        terminated = sum(self._required.values()) == 0

        reward = 100 if terminated else 0
        reward -= illegal_move_punishment
        reward -= 0.05

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, False, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self._render_human()

    def _render_human(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Factory Machines")
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw resource piles.
        resources = [self.res_out, self.res_steel, self.res_wood]
        for res in resources:
            pygame.draw.rect(
                canvas,
                self._pile_colours[res],
                pygame.Rect(
                    pix_square_size * self._piles[res],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent.
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _render_ansi(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_keys_to_action(self):
        return {
            'w': 0,
            'a': 1,
            's': 2,
            'd': 3,
            'g': 4,
            't': 5,
        }

    def _try_grab(self) -> bool:
        if self._carrying != 0:
            return False

        valid_res = [self.res_wood, self.res_steel]
        for res in valid_res:
            if np.array_equal(self._agent, self._piles[res]):
                # Successfully grabbed a resource.
                self._carrying = res
                return True

        return False

    def _try_drop(self) -> bool:
        if self._carrying != 0 and \
                np.array_equal(self._agent, self._piles[self.res_out]):
            self._required[self._carrying] = max(self._required[self._carrying] - 1, 0)
            self._carrying = 0
            return True

        return False
