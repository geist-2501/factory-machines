from typing import Tuple

import numpy as np


class Map:

    def __init__(self, layout, p, max_order_size) -> None:
        super().__init__()
        self.layout = layout
        self.p = p
        self.max_order_size = max_order_size

        output_loc, depot_locs, len_x, len_y = self._get_map_info()
        self.output_loc = output_loc
        self.depot_locs = depot_locs
        self.len_x = len_x
        self.len_y = len_y

        assert len(depot_locs) == len(p)

    def _get_map_info(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        output_loc = None
        depot_locs = []
        len_y = len(self.layout)
        len_x = len(self.layout[0])
        for y in range(len_y):
            for x in range(len_x):
                cell = self.layout[y][x]
                if cell == 'o':
                    # Get output loc.
                    output_loc = np.array([x, y], dtype=int)
                elif cell == 'd':
                    # Get depot locs.
                    depot_locs.append(np.array([x, y], dtype=int))

        return output_loc, np.array(depot_locs), len_x, len_y