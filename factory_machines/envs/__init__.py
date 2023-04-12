from gym.envs.registration import register

from factory_machines.envs.fm_env_multi import FactoryMachinesEnvMulti
from factory_machines.envs.fm_env_single import FactoryMachinesEnvSingle
from factory_machines.envs.orderworld_basic import OrderWorldBasic

register(
    id='FMMulti-v0',
    entry_point='factory_machines.envs:FactoryMachinesEnvMulti',
    max_episode_steps=300,
)

register(
    id='FMSingle-v0',
    entry_point='factory_machines.envs:FactoryMachinesEnvSingle',
    max_episode_steps=300,
)

register(
    id='DiscreteStochasticMDP-v0',
    entry_point='factory_machines.envs:DiscreteStochasticMDP',
    max_episode_steps=300,
)

register(
    id='GridWorld-v0',
    entry_point='factory_machines.envs:GridWorldBasic',
    max_episode_steps=300,
)

register(
    id='OrderWorld-v0',
    entry_point='factory_machines.envs:OrderWorldBasic',
    max_episode_steps=300,
)