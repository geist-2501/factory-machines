import math
from collections import defaultdict, deque
from typing import Tuple, List, Union, Optional, Deque

import heapq
import numpy as np

Coord = np.ndarray
HCoord = Tuple[int, int]  # Hashable coord.
Obs = np.ndarray
Bound = Tuple[Coord, Coord, Coord, Coord]


def manhatten_heuristic(start: HCoord, end: HCoord) -> int:
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


class SlamAstar:
    """Implementation of Simultaneous Localization and Mapping for A*."""

    up, left, down, right, no_op = range(5)

    def __init__(self, initial_obs: Obs):
        self.map = initial_obs
        len_y, len_x = initial_obs.shape
        self._origin = np.array([len_x // 2, len_y // 2])
        self.debug_mode = False

    def astar(self, start: Coord, end: Coord) -> Optional[List[HCoord]]:
        start = tuple(start)
        end = tuple(end)
        connections = {
            start: None
        }
        queue = [(0.0, start)]
        lengths = defaultdict(lambda: math.inf)
        lengths[start] = 0

        while queue:
            _, node = heapq.heappop(queue)
            if node == end:
                path = self._construct_path(connections, node, start)
                return path

            for neighbour in self._get_neighbours(node):
                path_length = lengths[node] + 1
                if path_length < lengths[neighbour]:
                    connections[neighbour] = node
                    lengths[neighbour] = path_length
                    f = path_length + manhatten_heuristic(neighbour, end)
                    heapq.heappush(queue, (f, neighbour))

        return None

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

        path = self.astar(start, end)

        if path is None or len(path) == 1:
            self._log(f"Could not find path from {start} to {end}")
            return self.no_op

        curr_coord = path[0]  # Will always start with the current coord.
        next_coord = path[1]

        return self._to_direction(curr_coord, next_coord)

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

    def _get_neighbours(self, coord: HCoord) -> List[HCoord]:
        offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        neighbours = list(map(lambda o: self._add_hcoord(coord, o), offsets))
        valid_neighbours = []
        for neighbour in neighbours:
            neighbour_coord = np.array(neighbour)
            if not self._is_oob(neighbour_coord) and self._get_map(neighbour_coord) != 1:
                valid_neighbours.append(neighbour)

        return valid_neighbours

    @staticmethod
    def _add_hcoord(a: HCoord, b: HCoord) -> HCoord:
        return a[0] + b[0], a[1] + b[1]

    @staticmethod
    def _construct_path(connections, node, start) -> List:
        path = [node]
        while node != start:
            node = connections[node]
            path.append(node)
        path.reverse()
        return path

    def _to_direction(self, a: HCoord, b: HCoord) -> int:
        a_x, a_y = a
        b_x, b_y = b
        d_x = b_x - a_x
        d_y = b_y - a_y

        if d_x == 0 and d_y == -1:
            return self.up
        elif d_x == 0 and d_y == 1:
            return self.down
        elif d_x == 1 and d_y == 0:
            return self.right
        elif d_x == -1 and d_y == 0:
            return self.left

        raise RuntimeError(f"Cannot convert ({d_x}, {d_y}) to unit direction.")

    def _log(self, message):
        if self.debug_mode:
            print(f"SLAM> {message}")

    def _print_map(self, points: List[Coord]):
        if not self.debug_mode:
            return

        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if self.map[y, x] == 1:
                    print("W", end="")
                elif (x, y) in points:
                    print("#", end="")
                else:
                    print(".", end="")
            print()