import gym
from factory_machines_env.wrappers.util import make_relative


class GridWorldFlattenRelativeWrapper(gym.ObservationWrapper):
    """
    Turns the dict observation from the FactoryMachinesEnv into a flat 1D list.
    Makes all locations relative.
    """
    def observation(self, obs):
        return [
            *obs["agent_loc"],
            *obs["agent_obs"],
            *make_relative(obs["agent_loc"], obs["checkpoints"]),
            *obs["goal"]
        ]
