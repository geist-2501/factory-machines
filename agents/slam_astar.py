from typing import Tuple

import numpy as np

Coord = np.ndarray
Obs = np.ndarray
Bound = Tuple[Coord, Coord, Coord, Coord]


class SlamAstar:
    """Implementation of Simultaneous Localization and Mapping for A*."""

    up, down, left, right, no_op = range(5)

    def __init__(self, initial_obs: Obs):
        self.map = initial_obs
        len_y, len_x = initial_obs.shape
        self._origin = np.array([len_x // 2, len_y // 2])

    def update(self, location: Coord, local_obs: Obs):
        """Update the understanding of the map."""
        bounds = self._get_bounds(location, local_obs)
        for bound in bounds:
            self._expand(bound)

        for map_coord, content in self._range_map_coords(location, local_obs):
            self._set_map(map_coord, content)

    def path_to(self, start: Coord, end: Coord) -> int:
        """Get the next direction in the path from start to end."""
        self._expand(end)
        # TODO
        return self.no_op

    def _is_oob(self, point: Coord) -> bool:
        """Check if `point` is out-of-bounds. Note that `point` should be in map-space."""
        x, y = self._origin + point
        len_y, len_x = self.map.shape

        return x < 0 or x >= len_x or y < 0 or y >= len_y

    def _expand(self, to: Coord):
        """Expand the local map to include the `to` coord."""
        if self._is_oob(to) is False:
            # Don't expand if we don't have to.
            return

        x, y = self._origin + to
        len_y, len_x = self.map.shape
        top_padding = abs(min(0, y))
        left_padding = abs(min(0, x))
        right_padding = max(len_x, x + 1) - len_x
        bottom_padding = max(len_y, y + 1) - len_y

        self._origin = self._origin + np.array([left_padding, top_padding])
        self.map = np.pad(self.map, ((top_padding, bottom_padding), (left_padding, right_padding)))

    @staticmethod
    def _range_map_coords(loc: Coord, local_obs: Obs):
        len_y, len_x = local_obs.shape
        offset = loc - np.array([len_x // 2, len_y // 2])
        for y in range(len(local_obs)):
            for x in range(len(local_obs[y])):
                yield offset + np.array([x, y]), local_obs[y, x]

    def _set_map(self, coord: Coord, value: int):
        # Convert to absolute space.
        x, y = coord + self._origin
        self.map[y, x] = value

    def _get_map(self, coord: Coord):
        # Convert to absolute space.
        x, y = coord + self._origin
        return self.map[y, x]

    @staticmethod
    def _get_bounds(center: Coord, local_obs: Obs) -> Bound:
        (len_x, len_y) = local_obs.shape
        assert len_x == len_y and len_x % 2 == 1, "Observation should be square and of odd length"

        bottom_right_offset = np.array([len_x // 2, len_y // 2])
        bottom_left_offset = bottom_right_offset * np.array([-1, 1])
        top_left_offset = bottom_right_offset * np.array([-1, -1])
        top_right_offset = bottom_right_offset * np.array([1, -1])

        return center + bottom_right_offset, \
            center + bottom_left_offset, \
            center + top_left_offset, \
            center + top_right_offset
