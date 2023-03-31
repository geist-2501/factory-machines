from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pygame
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame


class GridWorldBasic(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}
    maps: Dict[str, List] = {
        "0": [
            'c..c',
            '...w',
            'c..c',
        ]
    }

    colors = {
        'background': (255, 255, 255),
        'foreground': (50, 50, 50),
        'gridlines': (214, 214, 214),
        'text': (10, 10, 10),
        'agent': (43, 79, 255),
        'agent-light': (7, 35, 176),
    }

    up, left, down, right = range(4)

    def __init__(
            self,
            render_mode: Optional[str] = None,
            map_id="0",
    ) -> None:
        self._map = self.maps[map_id]

        self._checkpoints, self._len_x, self._len_y = self._get_map_info(self._map)
        self._goal = 0
        self._agent_loc = np.array([0, 0])
        self._num_checkpoints_left = 10

        self._last_action = 0
        self._last_reward = 0

        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(0, np.array([self._len_x, self._len_y]) - 1, shape=(2,), dtype=int),
                "agent_obs": spaces.Box(0, 1, shape=(9,), dtype=int),
                "checkpoints": spaces.Box(0, max(self._len_x, self._len_y), shape=(len(self._checkpoints) * 2,), dtype=int),
                "goal": spaces.Box(0, 1, shape=(len(self._checkpoints),))
            }
        )

        self.action_space = spaces.Discrete(5)  # Up, down, left, right.

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
        a_x, a_y = self._agent_loc
        for x in range(3):
            for y in range(3):
                map_x = a_x + x - 1
                map_y = a_y + y - 1
                if self._is_oob(map_x, map_y) or self._map[map_y][map_x] == 'w':
                    local_obs[y, x] = 1

        return {
            "agent_loc": self._agent_loc,
            "agent_obs": local_obs.flatten(),
            "checkpoints": self._checkpoints.flatten(),
            "goal": self._onehot_goal()
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self._agent_loc = np.array([0, 0])
        self._goal = 0
        self._num_checkpoints_left = 10

        obs = self._get_obs()

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Process actions.
        reward = 0

        if action < 4:
            # Action is a move op.
            direction = self._action_to_direction[action]
            new_pos = self._agent_loc + direction
            if self._is_oob(new_pos[0], new_pos[1]) or self._map[new_pos[1]][new_pos[0]] == 'w':
                reward += -1
            else:
                self._agent_loc = new_pos
        else:
            # Check depot drop off.
            if np.array_equal(self._agent_loc, self._checkpoints[self._goal]):
                self._goal = np.random.choice(len(self._checkpoints))
                self._num_checkpoints_left -= 1
                reward += 5
            else:
                reward += -1

        reward += -0.01

        # terminated = False
        terminated = self._num_checkpoints_left == 0

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        self._last_action = action
        self._last_reward = reward

        return obs, reward, terminated, False, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        len_x = self._len_x
        len_y = self._len_y
        cell_size = 64
        spacing = 8

        pygame.font.init()
        font = pygame.font.SysFont("monospace", 13)
        bottom_header = 20

        screen_width, screen_height = cell_size * len_x, cell_size * len_y + bottom_header

        pygame.init()
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill(self.colors["background"])

        # Add gridlines
        for x in range(len_x + 1):
            pygame.draw.line(
                self.screen,
                self.colors["gridlines"],
                (cell_size * x, 0),
                (cell_size * x, len_y * cell_size),
                width=3,
            )

        for y in range(len_y + 1):
            pygame.draw.line(
                self.screen,
                self.colors["gridlines"],
                (0, cell_size * y),
                (len_x * cell_size, cell_size * y),
                width=3,
            )

        # Draw checkpoints
        for checkpoint_num, checkpoint_loc in enumerate(self._checkpoints):
            checkpoint_text = f"C{checkpoint_num}"
            if checkpoint_num == self._goal:
                checkpoint_text += "-T"
            checkpoint_text_rect = font.render(checkpoint_text, True, self.colors["text"])
            self.screen.blit(checkpoint_text_rect, checkpoint_loc * cell_size)

        # Draw walls.
        for x in range(self._len_x):
            for y in range(self._len_y):
                if self._map[y][x] == 'w':
                    pygame.draw.rect(
                        self.screen,
                        self.colors["foreground"],
                        pygame.Rect(
                            (x * cell_size, y * cell_size),
                            (cell_size, cell_size)
                        )
                    )

        # Draw agent.
        pygame.draw.circle(
            self.screen,
            self.colors["agent"] if self._last_action != 4 else self.colors["agent-light"],
            (self._agent_loc + 0.5) * cell_size,
            cell_size / 3,
        )

        # Draw header.
        last_key = self.get_action_to_key(self._last_action)
        header_text_rect = font.render(f"A:{last_key} -> R:{self._last_reward}", True, self.colors["text"])
        self.screen.blit(header_text_rect, (0, cell_size * len_y))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def get_keys_to_action():
        return {
            'w': 0,
            'a': 1,
            's': 2,
            'd': 3,
            'g': 4,
        }

    def get_action_to_key(self, action):
        keys_to_action = self.get_keys_to_action()
        for key, key_action in keys_to_action.items():
            if action == key_action:
                return key

        raise RuntimeError(f"No matching key for action {action}")

    @staticmethod
    def _get_map_info(m: List) -> Tuple[np.ndarray, int, int]:
        checkpoints = []
        len_y = len(m)
        len_x = len(m[0])
        for y in range(len_y):
            for x in range(len_x):
                cell = m[y][x]
                if cell == 'c':
                    checkpoints.append(np.array([x, y], dtype=int))

        return np.array(checkpoints), len_x, len_y

    def _is_oob(self, x: int, y: int):
        return x < 0 or x >= self._len_x or y < 0 or y >= self._len_y

    def _onehot_goal(self):
        return [1 if i == self._goal else 0 for i in range(len(self._checkpoints))]