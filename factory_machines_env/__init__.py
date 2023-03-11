from gym.envs.registration import register
from factory_machines_env.wrappers import FactoryMachinesFlattenWrapper, FactoryMachinesFlattenRelativeWrapper, FactoryMachinesRelativeWrapper
from talos import register_wrapper

register(
    id='FMMulti-v0',
    entry_point='factory_machines_env.envs:FactoryMachinesEnvMulti',
    max_episode_steps=300,
)

register(
    id='FMSingle-v0',
    entry_point='factory_machines_env.envs:FactoryMachinesEnvSingle',
    max_episode_steps=300,
)

register(
    id='DiscreteStochasticMDP-v0',
    entry_point='factory_machines_env.envs:DiscreteStochasticMDP',
    max_episode_steps=300,
)

register_wrapper(
    id='FMFlatten',
    wrapper_factory=lambda outer: FactoryMachinesFlattenWrapper(outer)
)

register_wrapper(
    id='FMFlattenRel',
    wrapper_factory=lambda outer: FactoryMachinesFlattenRelativeWrapper(outer)
)

register_wrapper(
    id='FMRel',
    wrapper_factory=lambda outer: FactoryMachinesRelativeWrapper(outer)
)
