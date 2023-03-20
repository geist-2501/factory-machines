from typing import Dict, Tuple, Any

import numpy as np

from agents.slam_astar import SlamAstar
from talos import Agent


class FMSimpleHeuristicAgent(Agent):

    _up, _left, _right, _down, _grab = range(5)

    def __init__(self, obs, n_actions, device) -> None:
        super().__init__("Simple")

        assert type(obs) is dict, "Expected readable dictionary observation."
        local_obs = obs["agent_obs"].reshape(3, 3)
        raw_depot_locs = obs["depot_locs"]
        num_depots = len(raw_depot_locs) // 2
        self._depot_locs = np.array(raw_depot_locs).reshape(num_depots, 2)
        self._output_loc = obs["output_loc"]

        self._nav = SlamAstar(local_obs)

        self._last_idle = self._up

    def get_action(self, state, extra_state=None) -> Tuple[int, Any]:

        agent_obs = state["agent_obs"].reshape(3, 3)
        agent_loc = state["agent_loc"]
        depot_queues = state["depot_queues"]
        target_depot = extra_state

        self._nav.update(agent_loc, agent_obs)

        if target_depot is None or np.array_equal(agent_loc, self._output_loc):
            # Pick new target depot.
            if sum(depot_queues) == 0:
                return self._idle(), target_depot  # Do nothing.
            target_depot = self._depot_locs[np.argmax(depot_queues)]

        # Check if on target depot.
        if np.array_equal(agent_loc, target_depot):
            target_depot = self._output_loc
            return self._grab, target_depot

        # Otherwise, get next move direction.
        direction = self._nav.path_to(agent_loc, target_depot)
        if direction == self._nav.no_op:
            raise RuntimeError("Pathfinding failure.")

        return direction, target_depot

    def save(self) -> Dict:
        return {}

    def load(self, agent_data: Dict):
        pass

    def _idle(self):
        action = self._down if self._last_idle == self._up else self._up
        self._last_idle = action
        return action
