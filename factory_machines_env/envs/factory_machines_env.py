from typing import Optional, Union, List, Tuple

import gym
from gym import spaces
import pygame
import numpy as np
from gym.core import RenderFrame, ActType, ObsType


def _get_map_info(m: List[str]) -> Tuple[Tuple[int, int], List[Tuple[int, int]], int, int]:
    output_loc = None
    depot_locs = []
    for y in range(len(m)):
        for x in range(len(m[y])):
            cell = m[y][x]
            if cell == 'o':
                # Get output loc.
                output_loc = (x, y)
            elif cell == 'd':
                # Get depot locs.
                depot_locs.append((x, y))

    len_y = len(m)
    len_x = len(m[0])

    return output_loc, depot_locs, len_x, len_y


class FactoryMachinesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    maps = {
        1: [
            '.......',
            '.....1.',
            '.o...2.',
            '.....3.',
            '.......',
        ],
        2: [
            '.......',
            '...w.d.',
            '.o.w.d.',
            '...w.d.',
            '.......',
        ]
    }

    def __init__(self, render_mode: Optional[str] = None, map=1) -> None:
        self._map = self.maps[map]

        output_loc, depot_locs, len_x, len_y = _get_map_info(self._map)

        self._output_loc = output_loc
        self._depot_locs = depot_locs

        self._len_x = len_x
        self._len_y = len_y

        self._agent_loc = np.array(output_loc)
        self._agent_inv = np.array([] * len(depot_locs))
        self._depot_queues = np.array([] * len(depot_locs))

        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(0, np.array([len_x, len_y]) - 1, shape=(2,), dtype=int),
                "agent_obs": spaces.Box(0, 1, shape=(3, 3), dtype=int),
                "agent_inv": spaces.Box(0, 10, shape=(len(self._depot_locs),), dtype=int),
                "depot_locs": spaces.Box(0, np.array([len_x, len_y]) - 1, shape=(len(self._depot_locs), 2), dtype=int),
                "depot_queues": spaces.Box(0, 10, shape=(len(self._depot_locs),), dtype=int),
                "output_loc": spaces.Box(0, np.array([len_x, len_y]) - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(6)  # Up, down, left, right, grab.

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

        local_obs = np.zeros((3, 3))
        for x in range(-1, 2):
            for y in range(-1, 2):
                offset_x = self._agent_loc[0] + x
                offset_y = self._agent_loc[1] + y
                if self._is_oob(offset_x, offset_y):
                    local_obs[x, y] = 1

        return {
            "agent_loc": self._agent_loc,
            "agent_obs": local_obs,
            "agent_inv": self._agent_inv,
            "depot_locs": self._depot_locs,
            "depot_queues": self._depot_queues,
            "output_loc": self._output_loc,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        self._agent_loc = self._output_loc

        self._agent_inv = np.array([] * len(self._depot_locs))
        self._depot_queues = np.array([] * len(self._depot_locs))

        obs = self._get_obs()

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        illegal_move_punishment = 0
        if action < 4:
            # Action is a move op.
            direction = self._action_to_direction[action]
            new_pos = self._agent_loc + direction
            self._agent_loc = np.clip(new_pos, 0, [self._len_x - 1, self._len_y - 1])
        elif action == 4:
            # Action is a grab op.
            if self._try_grab() is False:
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


    def _is_oob(self, x: int, y: int):
        return x < 0 or x >= self._len_x or y < 0 or y >= self._len_y
