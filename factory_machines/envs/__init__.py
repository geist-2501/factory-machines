from factory_machines.envs.fm_env_multi import FactoryMachinesEnvMulti, fm_multi_graphing_wrapper
from factory_machines.envs.fm_env_multi_v2 import FactoryMachinesEnvMultiV2
from factory_machines.envs.fm_env_single import FactoryMachinesEnvSingle
from factory_machines.envs.orderworld_basic import OrderWorldBasic
from talos import register_env

register_env(
    env_id='FMMulti-v0',
    entry_point='factory_machines.envs:FactoryMachinesEnvMulti',
    graphing_wrapper=fm_multi_graphing_wrapper
)

register_env(
    env_id='FMMulti-alt-v0',
    entry_point='factory_machines.envs:FactoryMachinesEnvMultiV2',
    graphing_wrapper=fm_multi_graphing_wrapper
)

register_env(
    env_id='FMSingle-v0',
    entry_point='factory_machines.envs:FactoryMachinesEnvSingle',
)

register_env(
    env_id='DiscreteStochasticMDP-v0',
    entry_point='factory_machines.envs:DiscreteStochasticMDP',
)

register_env(
    env_id='GridWorld-v0',
    entry_point='factory_machines.envs:GridWorldBasic',
)

register_env(
    env_id='OrderWorld-v0',
    entry_point='factory_machines.envs:OrderWorldBasic',
)