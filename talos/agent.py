import time
from abc import ABC, abstractmethod

import gym


class Agent(ABC):
    """Base class for an agent for use in the Talos ecosystem."""

    @abstractmethod
    def get_action(self, state):
        """Request an action."""
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


def play_agent(agent: Agent, env: gym.Env, max_episode_steps=1000):
    s, _ = env.reset()
    total_ep_reward = 0
    for _ in range(max_episode_steps):
        action = agent.get_action(s)
        s, r, done, _, _ = env.step(action)
        total_ep_reward += r

        env.render()
        # plt.imshow(env.render())
        time.sleep(0.05)

        if done:
            env.close()
            break

    env.close()
    return total_ep_reward
