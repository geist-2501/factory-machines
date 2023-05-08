from typing import Union

import numpy as np

from factory_machines.envs import FactoryMachinesEnvMulti


class FactoryMachinesEnvMultiV2(FactoryMachinesEnvMulti):
    """
    Same as FactoryMachinesEnvMulti, except it has a smooth reward bonus for orders.
    """

    _max_age = 1
    _max_age_reward = 3
    _age_max_timesteps = 50

    def _sample_age_reward(self, age: int) -> float:
        compressed_age = self._get_age(age)
        reward_ratio = (self._max_age - compressed_age) / self._max_age

        return reward_ratio * self._max_age_reward

    def _get_age(self, order_age: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        compressed = (order_age * self._max_age) / self._age_max_timesteps
        return np.minimum(compressed, self._max_age)
