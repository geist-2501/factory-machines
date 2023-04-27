import math
from collections import deque
from typing import List, Tuple

import numpy as np
import pygame.draw
from pygame import Surface

Coord = np.ndarray
Colour = Tuple[int, int, int]


class RouteTracer:
    def __init__(self, k: int = None, max_width=8) -> None:
        self.route = deque(maxlen=k)
        self.pois = deque(maxlen=k)
        self.max_width = max_width

    def trace(self, coord: Coord, is_poi=False):
        """Add a location in the route tracer. Assumed to be after the previous trace invokation."""
        self.route.append(coord)
        self.pois.append(is_poi)

    def render(self, screen: Surface, cell_size: int, from_colour: Colour, to_colour: Colour):
        """Blit the route to the provided surface. Assumes an origin of (0,0)."""
        if len(self.route) < 2:
            return

        from_colour = pygame.Color(from_colour)
        to_colour = pygame.Color(to_colour)

        route = np.array(self.route) * cell_size + (cell_size // 2)
        current_coord = route[0]
        for n_coord, next_coord in enumerate(route[1:]):
            n_coord += 1  # It's actually ahead by 1 since we skip the first coord.
            progress = n_coord / len(route)
            colour = from_colour.lerp(to_colour, progress)
            width = max(math.ceil(self.max_width * progress), 1)
            pygame.draw.line(screen, colour, current_coord, next_coord, width=width)

            is_poi = self.pois[n_coord]
            if is_poi:
                pygame.draw.circle(screen, colour, next_coord, width + 2)

            current_coord = next_coord
