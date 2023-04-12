import gym

from factory_machines.wrappers.util import make_relative


class GridWorldRelativeWrapper(gym.ObservationWrapper):
    """
    Makes the dict observation from a FactoryMachines env have relative locations.
    """
    def observation(self, obs):
        return {
            **obs,
            "checkpoints": make_relative(obs["agent_loc"], obs["checkpoints"]),
        }