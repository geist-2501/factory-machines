from gym.envs.registration import register

register(
    id='factory_machines/FactoryMachinesMulti-v0',
    entry_point='factory_machines_env.envs:FactoryMachinesEnvMulti',
    max_episode_steps=300,
)

register(
    id='factory_machines/FactoryMachinesSingle-v0',
    entry_point='factory_machines_env.envs:FactoryMachinesEnvSingle',
    max_episode_steps=300,
)
