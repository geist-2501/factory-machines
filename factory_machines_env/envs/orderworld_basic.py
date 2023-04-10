import math
from typing import Optional, Union, List, Tuple, Dict

import gym
import numpy as np
import pygame
from gym import spaces
from gym.core import RenderFrame, ActType, ObsType

from factory_machines_env.envs.order_generators import GaussianOrderGenerator, OrderGenerator
from factory_machines_env.envs.pygame_utils import draw_lines

Coord = np.ndarray


class OrderWorldMap:

    def __init__(self, layout: List[str], p: List[float] = None) -> None:
        # Extract map depots and the lengths to each depot.
        self.n_depots, self.routes = self._get_layout_info(layout)

        if p is None:
            p = np.ones(self.n_depots - 1)

        assert len(p) == self.n_depots - 1, "Must have a probability for each depot!"
        self.p = p

    def _get_layout_info(self, layout: List[str]) -> Tuple[int, np.ndarray]:
        depots: List[Coord] = []
        output_depot: Optional[Coord] = None
        for y in range(len(layout)):
            for x in range(len(layout[y])):
                cell = layout[y][x]
                coord: Coord = np.array([x, y])
                if cell == 'o':
                    # Cell is output depot
                    output_depot = coord
                elif cell == 'd':
                    # Cell is regular depot.
                    depots.append(coord)

        assert output_depot is not None, "Map needs an output depot!"

        depots.append(output_depot)
        n_depots = len(depots)

        # Build route lengths.
        routes = np.ones((n_depots, n_depots))
        for i1, c1 in enumerate(depots):
            for i2, c2 in enumerate(depots):
                if i1 == i2:
                    continue
                routes[i1, i2] = self._euclidean_distance(c1, c2)

        return n_depots, routes

    @staticmethod
    def _euclidean_distance(c1: Coord, c2: Coord) -> float:
        dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        return round(dist, 1)


class OrderWorldBasic(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}
    maps: Dict[str, OrderWorldMap] = {
        "easy": OrderWorldMap([
            'o.d',
            '...',
            'd.d',
        ]),
        "easy-p": OrderWorldMap([
            'o.d',
            '...',
            'd.d',
        ], p=[0.5, 0.5, 0.8]),
        "medium": OrderWorldMap([
            '..o..',
            'd...d',
            '..d..',
            'd...d',
            '..d..',
        ]),
        "medium-2": OrderWorldMap([
            '...o...',
            '.d...d.',
            '.d...d.',
            '.d...d.',
            '.......',
        ])
    }

    _agent_cap = 10

    # Rewards.
    _travel_punishment = -0.1
    _item_dropoff_reward = 1  # The amount of reward for dropping off a needed item.
    _item_pickup_reward = 1
    _item_pickup_punishment = -0.5
    _reward_per_order = 10  # The amount of reward for dropping off a needed item.

    def __init__(
            self,
            render_mode: Optional[str] = None,
            map_id: str = "easy",
            num_orders: int = 10,
            generator: OrderGenerator = None
    ) -> None:
        super().__init__()

        self._map = self.maps[map_id]
        self._n_depots = self._map.n_depots
        self._n_item_depots = self._n_depots - 1

        self._order_generator = GaussianOrderGenerator(self._map.p, 4, generator=np.random.default_rng()) \
            if generator is None else generator

        self._current_depot = self._n_item_depots
        self._agent_inv = np.zeros(self._n_item_depots, dtype=int)

        num_orders = int(num_orders)
        self._n_orders_total = num_orders
        self._n_orders_left = num_orders
        self._open_orders: List[Tuple[int, np.ndarray]] = []
        self._timestep = 0

        # Metadata.
        self.action_space = spaces.Discrete(self._n_depots)
        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(0, 1, shape=(self._n_depots,), dtype=int),
                "agent_inv": spaces.Box(0, 10, shape=(self._n_item_depots,), dtype=int),
                "depot_queues": spaces.Box(0, 10, shape=(self._n_item_depots,), dtype=int),
            }
        )

        # Rendering.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            self._order_generator.set_seed(seed)

        self._timestep = 0
        self._current_depot = self._n_item_depots
        self._agent_inv = np.zeros(self._n_item_depots, dtype=int)
        self._n_orders_left = self._n_orders_total
        self._open_orders = []

        return self._get_obs(), {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        target_depot = action
        # Go to depot and grab an item from it. If the depot is not the current depot, incur a cost.
        reward = 0
        distance_travelled = self._map.routes[self._current_depot, target_depot]
        reward += distance_travelled * self._travel_punishment
        self._current_depot = target_depot
        if self._current_depot == self._n_depots - 1:
            # Depot is the output depot, complete orders.
            reward += self._depot_drop_off()
        else:
            # Depot is an item depot, grab item.
            reward += self._grab_item()

        # Process orders.
        should_create_order = self._order_generator.should_make_order(
            len(self._open_orders),
            elapsed_steps=round(distance_travelled)
        )
        if should_create_order and self._n_orders_left > 0:
            self._n_orders_left -= 1
            order = self._order_generator.make_order(self._n_item_depots)
            self._open_orders.append((self._timestep, order))

        terminated = self._n_orders_left == 0 and len(self._open_orders) == 0
        reward += 100 if terminated else 0

        self._timestep += 1

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        padding = 12

        pygame.font.init()
        font = pygame.font.SysFont("monospace", 15)

        font_width, font_height = font.size("GAP" + " SSS" * self._n_depots)
        screen_width = font_width + padding * 2
        screen_height = font_height * 10 + padding * 2

        pygame.init()
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        lines = [
            "   |" + ' '.join(["vvv" if i == self._current_depot else "   " for i in range(self._n_depots)]),
            "   |" + ' '.join([f"{f'D{i}':>3}" for i in range(self._n_depots - 1)]) + " OUT",
            "DEP|" + self._add_table_padding(self._get_depot_queues()),
            "INV|" + self._add_table_padding(self._agent_inv),
            "---|" + '-'.join([f"---" for i in range(self._n_depots - 1)]) + "----",
        ]

        for order in self._open_orders:
            order_t, order_items = order
            lines.append(f"{order_t:>3}|" + self._add_table_padding(order_items))

        draw_lines(lines, self.screen, (padding, padding), font, 0)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        super().close()
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self) -> Dict:
        return {
            "agent_loc": self._onehot(self._current_depot, self._n_depots),
            "agent_inv": self._agent_inv,
            "depot_queues": self._get_depot_queues(),
        }

    def _get_depot_queues(self):
        depot_queues = np.zeros(self._n_depots - 1, dtype=int)
        for order in self._open_orders:
            _, order_items = order
            depot_queues += order_items

        return depot_queues

    def _grab_item(self) -> float:
        queues = self._get_depot_queues()
        item_queue = queues[self._current_depot]
        if self._agent_inv[self._current_depot] >= self._agent_cap:
            return self._item_pickup_punishment

        elif self._agent_inv[self._current_depot] < item_queue:
            self._agent_inv[self._current_depot] += 1
            return self._item_pickup_reward

        elif self._agent_inv[self._current_depot] >= item_queue:
            self._agent_inv[self._current_depot] += 1
            return self._item_pickup_punishment

        raise RuntimeError("Shouldn't have gotten here lol")

    def _depot_drop_off(self) -> float:
        # Go through each open order and strike off items the agent holds.
        # Then, if an order has been completed, move it to the completed pile.
        reward = 0
        for i, order in enumerate(self._open_orders.copy()):
            order_t, order_items = order
            items_fulfilled = np.minimum(self._agent_inv, order_items)
            reward += sum(items_fulfilled) * self._item_dropoff_reward

            new_order = order_items - items_fulfilled
            self._open_orders[i] = (order_t, new_order)

            self._agent_inv -= items_fulfilled

            if sum(new_order) == 0:
                # Order is complete!
                order_age = self._timestep - order_t
                reward += self._reward_per_order

        # Remove complete orders.
        self._open_orders[:] = [order for order in self._open_orders if sum(order[1]) != 0]

        return reward

    @staticmethod
    def _add_table_padding(arr):
        return ' '.join(map(lambda i: f"{i:3}", arr))

    @staticmethod
    def get_keys_to_action():
        return {
            '1': 0,
            '2': 1,
            '3': 2,
            '4': 3,
            '5': 4,
            '6': 5,
            '7': 6,
            '8': 7,
        }

    @staticmethod
    def _onehot(category: int, n_categories: int) -> List[int]:
        return [1 if i == category else 0 for i in range(n_categories)]