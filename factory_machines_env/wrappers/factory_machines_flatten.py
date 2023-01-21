import gym
import numpy as np


class FactoryMachinesFlattenWrapper(gym.ObservationWrapper):
    """
    Turns the dict observation from the FactoryMachinesEnv into an ndarray.
    """
    def observation(self, observation):
        return (
            *observation["out_pile"],
            *observation["steel_pile"],
            *observation["wood_pile"],
            observation["req_wood"],
            observation["req_steel"],
            *observation["agent"],
            observation["carrying"],
        )
