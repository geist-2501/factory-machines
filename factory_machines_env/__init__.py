from gym.envs.registration import register

register(
    id='factory_machines/FactoryMachines-v0',
    entry_point='factory_machines_env.envs:FactoryMachinesEnv',
    max_episode_steps=300,
)
