from typing import Optional, Union, List, Tuple

import gym
import numpy as np
import pygame
from gym import spaces
from gym.core import RenderFrame, ActType, ObsType


def _get_map_info(m: List[str]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    output_loc = None
    depot_locs = []
    len_y = len(m)
    len_x = len(m[0])
    for y in range(len_y):
        for x in range(len_x):
            cell = m[y][x]
            if cell == 'o':
                # Get output loc.
                output_loc = np.array([x, y], dtype=int)
            elif cell == 'd':
                # Get depot locs.
                depot_locs.append(np.array([x, y], dtype=int))

    return output_loc, np.array(depot_locs), len_x, len_y


class FactoryMachinesEnvBase(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    maps = {
        1: [
            '.......',
            '.....d.',
            '.o...d.',
            '.....d.',
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

    def __init__(self, render_mode: Optional[str] = None, map_id=1) -> None:
        self._map = self.maps[map_id]

        output_loc, depot_locs, len_x, len_y = _get_map_info(self._map)

        self._output_loc = output_loc
        self._depot_locs = depot_locs
        self._num_depots = len(depot_locs)

        self._len_x = len_x
        self._len_y = len_y

        self._agent_loc = np.array(output_loc, dtype=int)
        self._agent_inv = np.zeros(self._num_depots, dtype=int)
        self._depot_queues = np.zeros(self._num_depots, dtype=int)

        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(0, np.array([len_x, len_y]) - 1, shape=(2,), dtype=int),
                "agent_obs": spaces.Box(0, 1, shape=(3, 3), dtype=int),
                "agent_inv": spaces.Box(0, 10, shape=(len(self._depot_locs),), dtype=int),
                "depot_locs": spaces.Box(0, max(len_x, len_y), shape=(len(self._depot_locs), 2), dtype=int),
                "depot_queues": spaces.Box(0, 10, shape=(len(self._depot_locs),), dtype=int),
                "output_loc": spaces.Box(0, max(len_x, len_y), shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(6)  # Up, down, left, right, grab.

        # Utility vectors for moving the agent.
        self._action_to_direction = {
            0: np.array([0, -1], dtype=int),  # w 0, -1
            1: np.array([-1, 0], dtype=int),  # a -1, 0
            2: np.array([0, 1], dtype=int),  # s 0, 1
            3: np.array([1, 0], dtype=int),  # d 1, 0
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Used for human friendly rendering.
        self.screen = None
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

        self._agent_inv = np.zeros(self._num_depots, dtype=int)
        self._depot_queues = np.zeros(self._num_depots, dtype=int)

        obs = self._get_obs()

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        # Process actions.
        grab_reward = 0
        if action < 4:
            # Action is a move op.
            direction = self._action_to_direction[action]
            new_pos = self._agent_loc + direction
            self._agent_loc = np.clip(new_pos, 0, [self._len_x - 1, self._len_y - 1])
        elif action == 4:
            # Action is a grab op.
            grab_reward = self._try_grab()

        # Check depot drop off.
        drop_off_reward = 0
        if np.array_equal(self._agent_loc, self._output_loc):
            agent_inv_inverse = 1 - self._agent_inv
            drop_off_reward = sum(self._depot_queues * self._agent_inv)
            self._depot_queues *= agent_inv_inverse  # Clear the queues of items the agent had.
            self._agent_inv = np.zeros(self._num_depots, dtype=int)

        reward = grab_reward + drop_off_reward

        obs = self._get_obs()
        info = {}

        return obs, reward, False, False, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        len_x = self._len_x
        len_y = self._len_y
        cell_size = 64
        spacing = 8

        header_size = cell_size * 2
        header_origin = (spacing, cell_size * len_y + spacing)

        screen_width, screen_height = cell_size * len_x, cell_size * len_y + header_size + spacing * 2

        black = (0, 0, 0)

        pygame.init()
        if self.screen is None:
            self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        font = pygame.font.Font(None, screen_height // 15)

        self.screen.fill((255, 255, 255))

        # Draw depots
        output_text = font.render("O", True, black)
        self.screen.blit(output_text, self._output_loc * cell_size)
        for depot_num, depot_loc in enumerate(self._depot_locs):
            depot_text = font.render("D" + str(depot_num), True, black)
            self.screen.blit(depot_text, depot_loc * cell_size)

        # Draw agent.
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            (self._agent_loc + 0.5) * cell_size,
            cell_size / 3,
        )

        # Add gridlines
        for x in range(len_x + 1):
            pygame.draw.line(
                self.screen,
                0,
                (cell_size * x, 0),
                (cell_size * x, len_y * cell_size),
                width=3,
            )

        for y in range(len_y + 1):
            pygame.draw.line(
                self.screen,
                0,
                (0, cell_size * y),
                (len_x * cell_size, cell_size * y),
                width=3,
            )

        inv_text = font.render("INV: " + str(self._agent_inv), True, black)
        inv_text_rect = self.screen.blit(inv_text, header_origin)

        depot_text = font.render("DEP: " + str(self._depot_queues), True, black)
        self.screen.blit(depot_text, (header_origin[0], inv_text_rect.bottom + spacing))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def get_keys_to_action(self):
        return {
            'w': 0,
            'a': 1,
            's': 2,
            'd': 3,
            'g': 4,
        }

    def _try_grab(self) -> int:
        """Try and add the current depot resource to the agent inventory.
        Returns reward if agent needed the resource, punishment if not."""
        for depot_num, depot_loc in enumerate(self._depot_locs):
            if not np.array_equal(self._agent_loc, depot_loc):
                continue

            # Agent is on a depot.
            if self._agent_inv[depot_num] != 0:
                # Agent is already holding material, administer punishment.
                return -1
            elif self._depot_queues[depot_num]:
                # Agent picks up resource,
                self._agent_inv[depot_num] = 1
                return 0

        return 0

    def _is_oob(self, x: int, y: int):
        return x < 0 or x >= self._len_x or y < 0 or y >= self._len_y
