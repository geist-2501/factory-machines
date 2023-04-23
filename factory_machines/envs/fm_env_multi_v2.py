from typing import Union

import numpy as np

from factory_machines.envs import FactoryMachinesEnvMulti


class FactoryMachinesEnvMultiV2(FactoryMachinesEnvMulti):
    _reward_per_order = 1
    _item_pickup_reward = 1
    _item_dropoff_reward = 1
    _item_pickup_punishment = -1
    _collision_punishment = -1
    _timestep_punishment = -0.5
    _episode_reward = 10

    _max_age = 1
    _max_age_reward = 1
    _age_max_timesteps = 50

    def _sample_age_reward(self, age: int) -> float:
        compressed_age = self._get_age(age)
        reward_ratio = (self._age_bands - compressed_age) / self._max_age

        return reward_ratio * self._max_age_reward

    def _get_age(self, order_age: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        compressed = (order_age * self._max_age) / self._age_max_timesteps
        return np.minimum(compressed, self._max_age)