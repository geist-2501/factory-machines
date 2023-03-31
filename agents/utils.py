from typing import Dict, List

import gym
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from talos import Agent

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

    def get_epsilon(self):
        return self._decay.get(self._tick)


class SuccessRateBasedDecay:
    def __init__(self, start_value, final_value, min_steps):
        self._decay_limit = StaticLinearDecay(start_value, final_value, min_steps)
        self.start_value = start_value
        self.final_value = final_value
        self._last_step = 0
        self.n_failed_attempts = 0
        self.n_successful_attempts = 0

    def get_success_rate(self) -> float:
        n_total_attempts = self.n_successful_attempts + self.n_failed_attempts
        if n_total_attempts == 0:
            return 0

        return self.n_successful_attempts / n_total_attempts

    def next(self, step: int, was_successful: bool) -> float:
        self._last_step = step
        if was_successful:
            self.n_successful_attempts += 1
        else:
            self.n_failed_attempts += 1

        return self.get_epsilon()

    def get_epsilon(self):
        inv_success_rate = (1 - self.get_success_rate()) * self.start_value
        minimum_decay = self._decay_limit.get(self._last_step)
        return max(inv_success_rate, minimum_decay)


class SuccessRateWithTimeLimitDecay:
    """Same as SuccessRateBasedDecay but considers actions beyond a time limit unsuccessful."""
    def __init__(self, start_value, final_value, min_steps, max_t: int):
        self._base_decay = SuccessRateBasedDecay(start_value, final_value, min_steps)
        self._max_t = max_t

    def get_success_rate(self) -> float:
        return self._base_decay.get_success_rate()

    def next(self, step: int, was_successful: bool, duration: int) -> float:
        was_successful &= duration <= self._max_t
        return self._base_decay.next(step, was_successful)

    def get_epsilon(self):
        return self._base_decay.get_epsilon()

def smoothen(data):
    return uniform_filter1d(data, size=30)


def evaluate(
        env: gym.Env,
        agent: Agent,
        n_episodes=1,
        max_episode_steps=10000
) -> float:
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
    return np.mean(total_ep_rewards).item()


def flatten(obs: Dict) -> List:
    return [
        *obs["agent_loc"],
        *obs["agent_obs"],
        *obs["agent_inv"],
        *obs["depot_locs"],
        *obs["depot_queues"],
        *obs["depot_ages"],
        *obs["output_loc"],
    ]


def parse_int_list(raw: str) -> List[int]:
    parts = raw.split(",")
    return list(map(lambda part: int(part), parts))
