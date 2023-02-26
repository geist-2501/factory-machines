import configparser
import time
from typing import List, Callable

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from talos.agent import Agent
from talos.error import *
from talos.registration import get_agent, get_wrapper
import talos.util as utils


def play_agent(
        agent: Agent,
        env: gym.Env,
        max_episode_steps=1000,
        wait_time: float = None
):
    s, _ = env.reset()
    reward_history = []
    info_history = []
    for _ in range(max_episode_steps):
        action = agent.get_action(s)
        s, r, done, _, info = env.step(action)
        reward_history.append(r)
        info_history.append(info)

        if wait_time:
            time.sleep(wait_time)

        if done:
            break

    env.close()
    return reward_history, info_history


def evaluate_agents(agents: List[Agent], agent_names: List[str], env_factory: Callable[[int], gym.Env]):
    reward_histories = []
    info_histories = []
    env_name = None
    for agent in agents:
        env = env_factory(0)
        env_name = type(env).__name__
        rewards, infos = play_agent(agent, env)
        reward_histories.append(rewards)
        info_histories.append(infos)

    reward_histories = utils.pad(reward_histories)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for i in range(len(agents)):
        axs[0].plot(np.cumsum(reward_histories[i]), label=agent_names[i])
    axs[1].bar(agent_names, np.sum(reward_histories, axis=1))

    axs[0].legend()
    axs[0].set_ylabel("Accumulated Reward")
    axs[1].set_ylabel("Final Reward")
    fig.suptitle(f"{' vs '.join(agent_names)} in {env_name}")
    fig.tight_layout()

    plt.show()


def load_config(config_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    result = config.read(config_path)
    if not result:
        raise ConfigNotFound

    return config


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_agent(env_factory, agent_name, device: str = get_device()):

    agent_factory, training_wrapper = get_agent(agent_name)
    env = env_factory(0)
    state, _ = env.reset()
    agent = agent_factory(
        len(state),
        env.action_space.n,
        device
    )

    return agent, training_wrapper


def create_env_factory(env_name, wrapper_name=None, render_mode=None):
    def env_factory(seed: int = None):
        env = gym.make(env_name, render_mode=render_mode).unwrapped

        if seed is not None:
            env.reset(seed=seed)

        if wrapper_name is not None:
            wrapper_factory = get_wrapper(wrapper_name)
            env = wrapper_factory(env)

        return env

    return env_factory
