import gym


class FactoryMachinesFlattenWrapper(gym.ObservationWrapper):
    """
    Turns the dict observation from the FactoryMachinesEnv into an ndarray.
    """
    def observation(self, observation):
        return [
            *observation["agent_loc"],
            *observation["agent_obs"].flatten(),
            *observation["agent_inv"],
            *observation["depot_locs"].flatten(),
            *observation["depot_queues"],
            *observation["output_loc"],
        ]
