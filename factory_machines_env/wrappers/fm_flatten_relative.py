import gym
import numpy as np


def make_relative(ref: np.ndarray, locs: np.ndarray):
    assert len(locs) % 2 == 0, "Locations must be a list of even length, in the form [x1, y1, x2, y2, ... ]"
    assert len(ref) == 2, "Reference must be an [x, y] coord"
    out = np.zeros(len(locs), dtype=int)
    for offset in range(len(locs) // 2):
        out[offset * 2] = locs[offset * 2] - ref[0]
        out[offset * 2 + 1] = locs[offset * 2 + 1] - ref[1]

    return out


class FactoryMachinesFlattenRelativeWrapper(gym.ObservationWrapper):
    """
    Turns the dict observation from the FactoryMachinesEnv into a flat 1D list.
    Makes all locations relative.
    """
    def observation(self, obs):
        return [
            *obs["agent_loc"],
            *obs["agent_obs"],
            *obs["agent_inv"],
            *make_relative(obs["agent_loc"], obs["depot_locs"]),
            *obs["depot_queues"],
            *make_relative(obs["agent_loc"], obs["output_loc"]),
        ]
