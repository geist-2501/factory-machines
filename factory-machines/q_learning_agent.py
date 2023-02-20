from collections import defaultdict
from time import sleep

import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:
    """
    A model-free QLearning agent, implemented according to Barto and Sutton.
    """

    def __init__(self, alpha: float, epsilon: float, discount: float, action_space: gym.spaces.Discrete):

        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.n_actions = action_space.n
        self.actions = range(self.n_actions)

        # Taken from the Practical_RL course.
        self.qvalues = defaultdict(lambda: defaultdict(lambda: 0))

    def _get_qvalue(self, state, action):
        return self.qvalues[state][action]

    def _set_qvalue(self, state, action, value):
        self.qvalues[state][action] = value

    def _get_actions(self):
        return [action for action in range(self.n_actions)]

    def get_value(self, state):
        """
        Compute the agent's estimate of V(s).
        """

        return max([self._get_qvalue(state, action) for action in self.actions])

    def update(self, state, action, reward, next_state):
        """
        Update the Q-values based on feedback from the environment.
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        v = self.get_value(next_state)
        old_q = self._get_qvalue(state, action)
        new_q = (1 - learning_rate) * old_q + learning_rate * (reward + gamma * v)

        self._set_qvalue(state, action, new_q)

    def get_optimal_action(self, state) -> int:
        """
        Compute the best action to take in a state.
        """

        qvals = np.array([self._get_qvalue(state, action) for action in self.actions], dtype=int)

        optimal_action = np.argmax(qvals)
        if isinstance(optimal_action, np.ndarray):
            return optimal_action[0]

        return optimal_action

    def get_epsilon_action(self, state) -> int:
        """
        Compute the action to take in the current state, including exploration according to an
        epsilon-greedy policy.
        """

        be_greedy = np.random.choice([False, True], p=[self.epsilon, 1 - self.epsilon])

        if be_greedy:
            return self.get_optimal_action(state)
        else:
            return np.random.choice(self.actions)


def train_q_learning_agent(env: gym.Env, agent: QLearningAgent, n_episodes=1000, episode_max_t=10**4):
    """
    Train a QLearning agent on an environment.
    """

    rewards = []
    for ep_t in range(n_episodes):
        s, _ = env.reset()
        total_reward_this_ep = 0

        for _ in range(episode_max_t):
            action = agent.get_epsilon_action(s)
            next_s, r, done, _, _ = env.step(action)
            agent.update(s, action, r, next_s)
            s = next_s

            total_reward_this_ep += r

            if done:
                break

        rewards.append(total_reward_this_ep)
        agent.epsilon *= 0.99

        if ep_t % 100 == 0:
            plt.title('eps = {:e}, mean reward = {:.1f}'.format(agent.epsilon, np.mean(rewards[-10:])))
            plt.plot(rewards)
            plt.pause(0.05)


def play_agent(env: gym.Env, agent: QLearningAgent, episode_max_t=10**4, wait_time_s=0.5):
    s, _ = env.reset()
    for _ in range(episode_max_t):
        a = agent.get_optimal_action(s)
        s, _, is_done, _, _ = env.step(a)
        env.render()

        sleep(wait_time_s)

        if is_done:
            break
