from typing import Tuple

import numpy as np

Coord = np.ndarray
Obs = np.ndarray
Bound = Tuple[Coord, Coord, Coord, Coord]


class SlamAstar:
    """Implementation of Simultaneous Location and Mapping for A*"""

    up, down, left, right, no_op = range(5)

    def __init__(self, initial_obs: Obs):
        self._map = initial_obs
        len_x, len_y = initial_obs.shape
        self._map_offset = np.ndarray([len_x // 2, -len_y // 2])

    def update(self, location: Coord, local_obs: Obs):
        """Update the understanding of the map."""
        bounds = self._get_bounds(location, local_obs)
        for bound in bounds:
            self._expand(bound)

        for absolute_coord, content in self._range_absolute_coords(location, local_obs):
            x, y = absolute_coord
            self._map[y, x] = content

    def path_to(self, start: Coord, end: Coord) -> int:
        """Get the next direction in the path from start to end."""
        self._expand(end)
        # TODO
        return self.no_op

    def _is_oob(self, point: Coord) -> bool:
        len_y, len_x = self._map.shape

        pass

    def _expand(self, to: Coord):
        """Expand the local map to include the `to` coord."""
        if self._is_oob(to) is False:
            # Don't expand if we don't have to.
            return
        # TODO
        pass

    def _range_absolute_coords(self, offset: Coord, local_obs: Obs):
        for y in range(len(local_obs)):
            for x in range(len(local_obs[y])):
                origin_to_point = offset + np.array([x, y])
                yield self._map_offset - origin_to_point, local_obs[y, x]

    @staticmethod
    def _get_bounds(center: Coord, local_obs: Obs) -> Bound:
        (len_x, len_y) = local_obs.shape
        assert len_x == len_y and len_x % 2 == 1, "Observation should be square and of odd length"

        top_right_offset = np.ndarray([len_x // 2, len_y // 2])
        top_left_offset = top_right_offset * np.ndarray([-1, 1])
        bottom_right_offset = top_right_offset * np.ndarray([1, -1])
        bottom_left_offset = top_left_offset * np.ndarray([-1, -1])

        return center + top_right_offset, \
            center + top_left_offset, \
            center + bottom_right_offset, \
            center + bottom_left_offset
