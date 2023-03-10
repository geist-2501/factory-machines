from typing import Dict, List

import gym
import numpy as np
from talos import Agent
from scipy.ndimage.filters import uniform_filter1d


try:
    import tkinter
    tkinter_available = True
except ModuleNotFoundError:
    tkinter_available = False


def can_graph():
    global tkinter_available
    return tkinter_available


class StaticLinearDecay:
    def __init__(self, start_value, final_value, max_steps):
        self.start_value = start_value
        self.final_value = final_value
        self.max_steps = max_steps

    def get(self, step):
        step = min(step, self.max_steps)
        upper = self.start_value * (self.max_steps - step)
        lower = self.final_value * step
        return (upper + lower) / self.max_steps


class MeteredLinearDecay:
    def __init__(self, start_value, final_value, max_steps):
        self._decay = StaticLinearDecay(start_value, final_value, max_steps)
        self._tick = 0

    def next(self):
        v = self._decay.get(self._tick)
        self._tick += 1
        return v


def smoothen(data):
    return uniform_filter1d(data, size=30)


def evaluate(
        env: gym.Env,
        agent: Agent,
        n_episodes=1,
        max_episode_steps=10000
):
    total_ep_rewards = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total_ep_reward = 0
        extra_state = None
        for _ in range(max_episode_steps):
            a, extra_state = agent.get_action(s, extra_state)
            s, r, done, _, _ = env.step(a)
            total_ep_reward += r

            if done:
                break

        total_ep_rewards.append(total_ep_reward)
    return np.mean(total_ep_rewards)


def flatten(obs: Dict) -> List:
    return [
        *obs["agent_loc"],
        *obs["agent_obs"].flatten(),
        *obs["agent_inv"],
        *obs["depot_locs"].flatten(),
        *obs["depot_queues"],
        *obs["output_loc"],
    ]


def parse_int_list(raw: str) -> List[int]:
    parts = raw.split(",")
    return list(map(lambda part: int(part), parts))
