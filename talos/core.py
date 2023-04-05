import configparser
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Callable, Any

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import print

from talos.agent import Agent
from talos.error import *
from talos.file import TalFile
from talos.registration import get_agent, get_wrapper, get_agent_graphing


def play_agent(
        agent: Agent,
        env: gym.Env,
        max_episode_steps=1000,
        wait_time: float = None
):
    obs, _ = env.reset()
    env.render()
    extra_state = None
    reward_history = []
    info_history = []
    for _ in range(max_episode_steps):
        action, extra_state = agent.get_action(obs, extra_state)
        next_obs, r, done, _, info = env.step(action)

        # print(agent.get_intrinsic_reward(obs, action, next_obs, extra_state))

        # Some agents require extra processing (looking at you, h-DQN).
        extra_state = agent.post_step(obs, action, next_obs, extra_state)

        obs = next_obs

        reward_history.append(r)
        info_history.append(info)

        if env.render_mode != "human":
            if env.render_mode == "rgb_array":
                plt.imshow(env.render())
                plt.pause(0.01)
            else:
                env.render()

        if wait_time:
            time.sleep(wait_time)

        if done:
            break

    env.close()
    return reward_history, info_history


def evaluate_agents(loaded_agents: List[Dict], max_episode_timesteps=1000, n_episodes=3):
    rewards = []
    final_infos = []
    for loaded_agent in loaded_agents:
        agent = loaded_agent["agent"]
        env_factory = loaded_agent["env_factory"]
        env = env_factory(0)
        print(f"Agent {loaded_agent['agent_name']}")
        total_reward, final_info = evaluate_agent(env, agent, n_episodes, max_episode_timesteps, verbose=True)
        rewards.append(total_reward)
        final_infos.append(final_info)

    return rewards, final_infos


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
        state,
        env.action_space.n,
        device,
    )

    return agent, training_wrapper


def create_env_factory(env_name, wrapper_name=None, render_mode='rgb_array', env_args=None, base_seed: int = 0):
    if env_args is None:
        env_args = {}

    def env_factory(seed: int = None):
        env = gym.make(env_name, render_mode=render_mode, **env_args).unwrapped

        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset(seed=base_seed)

        if wrapper_name is not None:
            wrapper_factory = get_wrapper(wrapper_name)
            env = wrapper_factory(env)

        return env

    return env_factory


def graph_agent(agent_id: str, artifacts: Dict):
    graphing_wrapper = get_agent_graphing(agent_id)
    graphing_wrapper(artifacts)
    plt.show()


def evaluate_agent(
        env: gym.Env,
        agent: Agent,
        n_episodes=1,
        max_episode_steps=10000,
        verbose=False
) -> Tuple[float, Dict[str, float]]:
    total_ep_rewards = []
    total_infos = defaultdict(lambda: [])
    for ep_num in range(n_episodes):
        obs, _ = env.reset()
        total_ep_reward = 0
        extra_state = None
        done = False
        for _ in range(max_episode_steps):
            action, extra_state = agent.get_action(obs, extra_state)
            next_obs, reward, done, _, info = env.step(action)
            total_ep_reward += reward

            # Some agents require extra processing (looking at you, h-DQN).
            extra_state = agent.post_step(obs, action, next_obs, extra_state)

            obs = next_obs

            if done:
                for k, v in info.items():
                    total_infos[k].append(v)

                if verbose:
                    print(f"Episode {ep_num} | R:{total_ep_reward}, I:{info}")
                break

        if not done and verbose:
            print("[red]Agent did not complete.[/]")

        total_ep_rewards.append(total_ep_reward)

    # Take average over total infos.
    total_infos_mean = {}
    for info, history in total_infos.items():
        total_infos_mean[info] = np.mean(history).item()

    return np.mean(total_ep_rewards).item(), total_infos_mean
