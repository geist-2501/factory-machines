from collections import deque
from typing import Tuple

import pygame


class History:
    def __init__(self, size=4):
        self.size = 4
        self._logs = deque(maxlen=size)

    def log(self, message: str):
        self._logs.append(message)

    def render(
            self,
            font: pygame.font.Font,
            color,
            width=300
    ) -> pygame.Surface:
        _, font_height = font.size("test")
        surface = pygame.Surface((width, font_height * self.size))
        surface.fill((255, 255, 255))
        y_offset = 0
        for entry in self._logs:
            history_text = font.render(entry, True, color)
            history_text_rect = surface.blit(history_text, (0, y_offset))
            y_offset = history_text_rect.bottom

        return surface
