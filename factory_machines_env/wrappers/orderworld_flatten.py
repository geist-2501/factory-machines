import gym


class OrderWorldFlattenWrapper(gym.ObservationWrapper):
    """
    Turns the dict observation from the FactoryMachinesEnv into a flat 1D list.
    Makes all locations relative.
    """
    def observation(self, obs):
        return [
            *obs["agent_loc"],
            *obs["agent_inv"],
            *obs["depot_queues"],
        ]
