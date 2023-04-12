from typing import Dict, Tuple, Any, Optional

import numpy as np

from factory_machines.agents.heuristics.slam_astar import SlamAstar
from talos import Agent

Coord = np.ndarray
Order = np.ndarray


class AgentState:
    def __init__(self, order: Order, target_index: Optional[int]) -> None:
        self.order = order
        self.target_index = target_index

    def num_required_from_target(self):
        return self.order[self.target_index]

    def decrement_target(self):
        self.order[self.target_index] -= 1


class NNAgent(Agent):

    _up, _left, _right, _down, _grab = range(5)

    def __init__(self, obs, n_actions, device) -> None:
        super().__init__("Nearest-Neighbour")

        assert type(obs) is dict, "Expected readable dictionary observation."
        local_obs = obs["agent_obs"].reshape(3, 3)
        raw_depot_locs = obs["depot_locs"]
        num_depots = len(raw_depot_locs) // 2
        self._depot_locs = np.array(raw_depot_locs).reshape(num_depots, 2)
        self._all_locs = np.append(self._depot_locs, [obs["output_loc"]], axis=0)
        self._output_idx = num_depots

        self._nav = SlamAstar(local_obs)

        self._last_idle = self._up

    def get_action(self, obs, extra_state=None) -> Tuple[int, Any]:
        agent_obs = obs["agent_obs"].reshape(3, 3)
        agent_loc = obs["agent_loc"]
        depot_queues = obs["depot_queues"]
        agent_state = extra_state

        self._nav.update(agent_loc, agent_obs)

        if sum(depot_queues) == 0:
            return self._idle(), agent_state

        # If no current state, take snapshot.
        if agent_state is None:
            agent_state = self._make_state(agent_loc, depot_queues)

        target_loc = self._get_target_loc(agent_state)
        if np.array_equal(agent_loc, target_loc):
            # On depot.
            if agent_state.target_index == self._output_idx:
                # On output depot.
                agent_state = self._make_state(agent_loc, depot_queues)
            elif agent_state.num_required_from_target() > 0:
                # Grab resource from target.
                agent_state.decrement_target()
                return self._grab, agent_state
            else:
                # Find new target.
                agent_state.target_index = self._get_next_target_index(agent_loc, agent_state.order)

        target_loc = self._get_target_loc(agent_state)

        # Otherwise, get next move direction.
        direction = self._nav.path_to(agent_loc, target_loc)
        if direction == self._nav.no_op:
            raise RuntimeError("Pathfinding failure.")

        return direction, agent_state

    def _make_state(self, agent_loc, depot_queues):
        order = np.copy(depot_queues)
        agent_state = AgentState(
            order=order,
            target_index=self._get_first_target_index(agent_loc, order)
        )
        return agent_state

    def save(self) -> Dict:
        return {}

    def load(self, agent_data: Dict):
        pass

    def _get_target_loc(self, agent_state):
        return self._all_locs[agent_state.target_index]

    def _get_first_target_index(self, agent_loc: Coord, order: Order) -> int:
        """Pick the depot from the order to start with."""
        return self._get_next_target_index(agent_loc, order)

    def _get_next_target_index(self, agent_loc: Coord, order: Order) -> int:
        """Get the next depot to visit in the order."""
        if sum(order) == 0:
            return self._output_idx

        distances = [self._nav.distance(agent_loc, self._depot_locs[idx]) if quant > 0 else np.inf
                     for idx, quant in enumerate(order)]
        return np.argmin(distances).item()

    def _idle(self):
        action = self._down if self._last_idle == self._up else self._up
        self._last_idle = action
        return action
