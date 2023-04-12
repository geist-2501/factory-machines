import gym


class FactoryMachinesFlattenWrapper(gym.ObservationWrapper):
    """
    Turns the dict observation from the FactoryMachinesEnv into a flat 1D list.
    """
    def observation(self, observation):
        return [
            *observation["agent_loc"],
            *observation["agent_obs"],
            *observation["agent_inv"],
            *observation["depot_locs"],
            *observation["depot_queues"],
            *observation["output_loc"],
        ]
