import configparser
from typing import Callable, Dict

import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from factory_machines.utils import LinearDecay
from factory_machines.replay_buffer import ReplayBuffer
from talos import Agent


class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions)
        )

    def forward(self, states):
        return self.net(states)


class DQNAgent(Agent):
    def __init__(self, obs_size, n_actions, epsilon=0, gamma=0.99, device='cpu'):
        super().__init__("DQN")
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.device = device

        # Init both networks.
        self.net = DQN(obs_size, n_actions).to(device)
        self.target_net = DQN(obs_size, n_actions).to(device)
        self.target_net.load_state_dict(self.net.state_dict())

    def forward(self, state):
        return self.net(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def compute_loss(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            is_done: np.ndarray
    ) -> torch.Tensor:
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        is_done = torch.tensor(
            is_done.astype('uint8'),
            device=self.device,
            dtype=torch.uint8,
        )

        # get q-values for all actions in current states
        predicted_qvalues = self.net(states)  # shape: [batch_size, n_actions]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.target_net(next_states)  # shape: [batch_size, n_actions]

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]  # shape: [batch_size]

        # compute V*(next_states) using predicted next q-values
        next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        # you can multiply next state values by is_not_done to achieve this.
        target_qvalues_for_actions = rewards + self.gamma * next_state_values
        target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

        return loss

    def get_action(self, state):
        return self.get_optimal_actions(np.array([state]))[0]

    def get_epsilon_actions(self, states: np.ndarray):
        """Pick actions according to an epsilon greedy strategy."""
        return self._get_actions(states, self.epsilon)

    def get_optimal_actions(self, states: np.ndarray):
        """Pick actions according to a greedy strategy."""
        return self._get_actions(states, 0)

    def _get_actions(self, states: np.ndarray, epsilon: float):
        """Pick actions according to an epsilon greedy strategy."""
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        qvalues = self.forward(states)

        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1).cpu()

        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

    def parameters(self):
        return self.net.parameters()

    def save(self) -> Dict:
        return self.net.state_dict()

    def load(self, agent_data: Dict):
        self.net.load_state_dict(agent_data)
        self.target_net.load_state_dict(agent_data)


def _play_into_buffer(
        env: gym.Env,
        agent: DQNAgent,
        buffer: ReplayBuffer,
        state,
        n_steps=1,
):
    s = state

    for _ in range(n_steps):
        action = agent.get_epsilon_actions(np.array([s]))[0]
        sp1, r, done, _, _ = env.step(action)
        buffer.add(s, action, r, sp1, done)
        s = sp1
        if done:
            env.reset()

    return s


def _evaluate(
        env: gym.Env,
        agent: DQNAgent,
        n_episodes=1,
        max_episode_steps=10000
):
    mean_ep_rewards = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total_ep_reward = 0
        for _ in range(max_episode_steps):
            action = agent.get_optimal_actions(np.array([s]))[0]
            s, r, done, _, _ = env.step(action)
            total_ep_reward += r

            if done:
                break

        mean_ep_rewards.append(total_ep_reward)
    return np.mean(mean_ep_rewards)


# timesteps_per_epoch = 1
# batch_size = 32
# total_steps = 4 * 10**4
# decay_steps = 1 * 10**4
#
# opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
#
# init_epsilon = 1
# final_epsilon = 0.1
#
# loss_freq = 20
# refresh_target_network_freq = 100
# eval_freq = 1000
#
# max_grad_norm = 5000

def train_dqn_agent(
        env_factory: Callable[[int], gym.Env],
        agent: DQNAgent,
        opt: torch.optim.Optimizer,
        epsilon_decay: LinearDecay = LinearDecay(1, 0.1, 1 * 10 ** 4),
        max_steps: int = 4 * 10 ** 4,
        timesteps_per_epoch=1,
        batch_size=30,
        update_target_net_freq=100,
        evaluation_freq=1000,
        replay_buffer_size=10**4
):
    env = env_factory(0)
    state, _ = env.reset()

    # Create buffer and fill it with experiences.
    buffer = ReplayBuffer(replay_buffer_size)
    for _ in range(100):
        state = _play_into_buffer(env, agent, buffer, state, n_steps=10**2)
        if len(buffer) >= replay_buffer_size:
            break

    state, _ = env.reset()

    mean_reward_history = []
    for step in trange(0, max_steps):

        agent.epsilon = epsilon_decay.get(step)

        state = _play_into_buffer(env, agent, buffer, state, timesteps_per_epoch)

        (s, a, r, s_dash, is_done) = buffer.sample(batch_size)

        loss = agent.compute_loss(s, a, r, s_dash, is_done)

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % update_target_net_freq == 0:
            # Load agent weights into target_network
            agent.update_target_net()

        if step % evaluation_freq == 0:
            mean_reward_history.append(_evaluate(
                env_factory(step), agent, n_episodes=3, max_episode_steps=1000)
            )
            plt.title('eps = {:e}, mean reward = {:.1f}'.format(agent.epsilon, np.mean(mean_reward_history[-10:])))
            plt.plot(mean_reward_history)
            plt.pause(0.05)


def dqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],
        agent: DQNAgent,
        dqn_config: configparser.SectionProxy
):
    train_dqn_agent(
        env_factory=env_factory,
        agent=agent,
        opt=torch.optim.NAdam(agent.parameters(), lr=dqn_config.getfloat("learning_rate")),
        epsilon_decay=LinearDecay(
            start_value=dqn_config.getfloat("init_epsilon"),
            final_value=dqn_config.getfloat("final_epsilon"),
            max_steps=dqn_config.getint("decay_steps")
        ),
        max_steps=dqn_config.getint("total_steps"),
        timesteps_per_epoch=dqn_config.getint("timesteps_per_epoch"),
        batch_size=dqn_config.getint("batch_size"),
        update_target_net_freq=dqn_config.getint("refresh_target_network_freq"),
        evaluation_freq=dqn_config.getint("eval_freq"),
        replay_buffer_size=dqn_config.getint("replay_buffer_size")
    )
