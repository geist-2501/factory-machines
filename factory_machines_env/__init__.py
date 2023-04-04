from gym.envs.registration import register
from factory_machines_env.wrappers import \
    FactoryMachinesFlattenWrapper, \
    FactoryMachinesFlattenRelativeWrapper, \
    FactoryMachinesRelativeWrapper, \
    ArrayWrapper, \
    GridWorldFlattenRelativeWrapper, \
    GridWorldRelativeWrapper, \
    OrderWorldFlattenWrapper
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

register(
    id='GridWorld-v0',
    entry_point='factory_machines_env.envs:GridWorldBasic',
    max_episode_steps=300,
)

register(
    id='OrderWorld-v0',
    entry_point='factory_machines_env.envs:OrderWorldBasic',
    max_episode_steps=300,
)

register_wrapper(
    wrapper_id='FMFlatten',
    wrapper_factory=lambda outer: FactoryMachinesFlattenWrapper(outer)
)

register_wrapper(
    wrapper_id='GWFlattenRel',
    wrapper_factory=lambda outer: GridWorldFlattenRelativeWrapper(outer)
)

register_wrapper(
    wrapper_id='GWRel',
    wrapper_factory=lambda outer: GridWorldRelativeWrapper(outer)
)

register_wrapper(
    wrapper_id='FMFlattenRel',
    wrapper_factory=lambda outer: FactoryMachinesFlattenRelativeWrapper(outer)
)

register_wrapper(
    wrapper_id='FMRel',
    wrapper_factory=lambda outer: FactoryMachinesRelativeWrapper(outer)
)

register_wrapper(
    wrapper_id='Array',
    wrapper_factory=lambda outer: ArrayWrapper(outer)
)

register_wrapper(
    wrapper_id='OWFlatten',
    wrapper_factory=lambda outer: OrderWorldFlattenWrapper(outer)
)