from talos import register_wrapper
from factory_machines.wrappers.array_wrapper import ArrayWrapper
from factory_machines.wrappers.fm_flatten import FactoryMachinesFlattenWrapper
from factory_machines.wrappers.fm_flatten_relative import FactoryMachinesFlattenRelativeWrapper
from factory_machines.wrappers.fm_relative import FactoryMachinesRelativeWrapper
from factory_machines.wrappers.gridworld_flatten_relative import GridWorldFlattenRelativeWrapper
from factory_machines.wrappers.gridworld_relative import GridWorldRelativeWrapper
from factory_machines.wrappers.orderworld_flatten import OrderWorldFlattenWrapper

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