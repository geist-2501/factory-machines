from typing import Optional, Union, List, Tuple

import gym
import numpy as np
import pygame
from gym.core import RenderFrame, ActType, ObsType
from gym import spaces

from factory_machines.envs.pygame_utils import draw_lines


class DiscreteStochasticMDP(gym.Env):

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}
    _left = 0
    _right = 1

    def __init__(self, render_mode: Optional[str] = None, is_deterministic: str = "False"):
        self._is_deterministic = is_deterministic == "True"  # lol.
        self._num_states = 6
        self.observation_space = spaces.Box(0, 1, shape=(self._num_states,), dtype=int)
        self.action_space = spaces.Discrete(2)  # Left, right.

        self._current_state = 1
        self._reward = 0.01
        self._last_action = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Used for human friendly rendering.
        self.screen = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        self._current_state = 1
        self._reward = 0.01

        return self._get_obs(), {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        self._last_action = action
        if action == self._left:
            # On left, move left.
            self._move_left()
        elif action == self._right:
            # On right, roll dice and move right on result > 0.5, left otherwise.
            if self._is_deterministic or self.np_random.random() > 0.5:
                self._move_right()
            else:
                self._move_left()

        # If agent goes into the last state, give a bigger reward on completion.
        if self._current_state == (self._num_states - 1):
            self._reward = 1

        terminated = self._current_state == 0
        reward = self._reward if terminated else 0
        obs = self._get_obs()

        return obs, reward, terminated, False, {}

    def _move_left(self):
        self._current_state = max(0, self._current_state - 1)

    def _move_right(self):
        self._current_state = min(self._num_states - 1, self._current_state + 1)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "ansi":
            return self._render_ansi()
        else:
            return self._render_rgb_array()

    def _render_ansi(self):
        pointer, states = self._get_lines()
        lines = f"{states}\n{pointer}"
        print(lines)
        if self._last_action is not None:
            print(f"Last action: {self._last_action} - current reward {self._reward}\n")
        return lines

    def _get_lines(self):
        states = ' '.join([f"S{i}" for i in range(self._num_states)])
        pointer = ' '.join(["^^" if i == self._current_state else "  "
                            for i in range(self._num_states)])
        return pointer, states

    def _render_rgb_array(self):

        padding = 12

        pygame.font.init()
        font = pygame.font.SysFont("monospace", 15)

        font_width, font_height = font.size(" " + "SS " * self._num_states)
        screen_width = font_width + padding * 2
        screen_height = font_height * 3 + padding * 2

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((screen_width, screen_height))

        self.screen.fill((255, 255, 255))
        pointer, states = self._get_lines()
        draw_lines([states, pointer, f"R: {self._reward}"], self.screen, (padding, padding), font, color=0)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _get_obs(self):
        return self._onehot(self._current_state, self._num_states)

    @staticmethod
    def _onehot(category: int, n_categories: int):
        v = np.zeros(n_categories)
        v[category] = 1
        return v

    @staticmethod
    def get_keys_to_action():
        return {
            'a': 0,
            'd': 1,
        }
