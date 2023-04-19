from factory_machines.envs import FactoryMachinesEnvMulti


class FactoryMachinesEnvMultiV2(FactoryMachinesEnvMulti):
    _reward_per_order = 1
    _item_pickup_reward = 1
    _item_dropoff_reward = 1
    _item_pickup_punishment = -1
    _collision_punishment = -1
    _timestep_punishment = -0.5
    _episode_reward = 10

    _age_bands = 100
    _max_age_reward = 1
    _age_max_timesteps = 50
