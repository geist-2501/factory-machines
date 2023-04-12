import gym


class ArrayWrapper(gym.ObservationWrapper):
    """
    Wraps an observation in an array.
    Useful for environments like Taxi, which just has an int as an observation.
    """
    def observation(self, obs):
        return [obs]
