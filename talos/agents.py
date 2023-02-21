from typing import Callable

import gym
import configparser
from rich import print
import torch.optim

from factory_machines import DQNAgent, train_dqn_agent, LinearDecay


def resolve_agent(agent_name: str):
    match agent_name:
        case "DQN":
            return dqn_training_wrapper
        case _:
            return None


def dqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],
        c: configparser.ConfigParser
):
    config = c['dqn']
    env = env_factory(0)
    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n
    )

    try:
        train_dqn_agent(
            env_factory=env_factory,
            agent=agent,
            opt=torch.optim.NAdam(agent.parameters(), lr=config.getfloat("learning_rate")),
            epsilon_decay=LinearDecay(
                start_value=config.getfloat("init_epsilon"),
                final_value=config.getfloat("final_epsilon"),
                max_steps=config.getint("decay_steps")
            ),
            max_steps=config.getint("total_steps"),
            timesteps_per_epoch=config.getint("timesteps_per_epoch"),
            batch_size=config.getint("batch_size"),
            update_target_net_freq=config.getint("refresh_target_network_freq"),
            evaluation_freq=config.getint("eval_freq"),
            replay_buffer_size=config.getint("replay_buffer_size")
        )
    except KeyboardInterrupt:
        print("[bold red]Training interrupted[/bold red], saving agent to disk...")

    agent.save("dqn_params.pt")
