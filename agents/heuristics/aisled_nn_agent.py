import numpy as np

from agents.heuristics.nn_agent import NNAgent, Coord, Order


class AisledNNAgent(NNAgent):
    def _get_first_target_index(self, agent_loc: Coord, order: Order):
        first_idx = np.argmin(order > 0).item()
        last_idx = np.argmax(order > 0).item()
        first_dist = self._nav.distance(agent_loc, self._depot_locs[first_idx])
        last_dist = self._nav.distance(agent_loc, self._depot_locs[last_idx])

        return first_idx if first_dist < last_dist else last_idx