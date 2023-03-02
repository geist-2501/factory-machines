from typing import Dict, Callable, Tuple, Any
from operator import itemgetter

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from factory_machines.dqn_agent import DQN
from factory_machines.replay_buffer import ReplayBuffer
from talos import Agent


class HDQNAgent(Agent):

    def __init__(
            self,
            obs_size: int,
            n_goals: int,
            n_actions: int,
            gamma: float = 0.99,
            device: str = 'cpu'
    ) -> None:
        super().__init__("h-DQN")

        self.device = device
        self.gamma = gamma

        self.eps1 = 1
        self.eps2 = np.ones(n_goals)

        self.n_goals = n_goals

        self.d1 = ReplayBuffer(10**4)
        self.d2 = ReplayBuffer(10**4)

        self.meta_cont_net = DQN(obs_size, n_goals).to(device)  # Meta-controller net / Q2.
        self.meta_cont_net_fixed = DQN(obs_size, n_goals).to(device)  # Meta-controller fixed net.
        self.cont_net = DQN(n_goals, n_actions).to(device)  # Controller net / Q1.
        self.cont_net_fixed = DQN(n_goals, n_actions).to(device)  # Controller fixed net.

    def forward(self, state):
        # TODO this is non-trivial.
        pass

    def compute_td_loss(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            is_done: np.ndarray,
            net: DQN,
            net_fixed: DQN
    ):
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        is_done = torch.tensor(
            is_done.astype('bool'),
            device=self.device,
            dtype=torch.bool,
        )

        # get q-values for all actions in current states
        predicted_qvalues = net(states)  # shape: [batch_size, n_actions]

        # compute q-values for all actions in next states
        predicted_next_qvalues = net_fixed(next_states)  # shape: [batch_size, n_actions]

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
        loss = F.mse_loss(target_qvalues_for_actions, predicted_qvalues_for_actions)

        return loss

    def get_action(self, state: np.ndarray, extra_state=None) -> Tuple[int, Any]:
        """Get the optimal action given a state."""
        goal = extra_state
        meta_controller_obs = np.expand_dims(state, axis=0)
        if goal is None:
            goal = self.get_epsilon(meta_controller_obs, 0, self.meta_cont_net)[0]

        controller_obs = np.append(meta_controller_obs, goal)
        action = self.get_epsilon(controller_obs, epsilon=0, net=self.cont_net)[0]

        return action, goal

    def get_epsilon_action(self, states: np.ndarray):
        return self.get_epsilon(states, self.)

    def get_epsilon(self, states: np.ndarray, epsilon: float, net: DQN):
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        net.eval()
        with torch.no_grad():
            qvalues = net(states)
        net.train()

        if len(states.shape) == 1:
            # Single version:
            n_actions = qvalues.shape[0]
            should_explore = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
            if should_explore:
                return np.random.choice(n_actions)
            else:
                qvalues.argmax().cpu()

        elif len(states.shape) == 2:
            # Batch version
            batch_size, n_actions = qvalues.shape

            random_actions = np.random.choice(n_actions, size=batch_size)
            best_actions = qvalues.argmax(axis=-1).cpu()

            should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
            return np.where(should_explore, random_actions, best_actions)

    def save(self) -> Dict:
        return {
            "meta_cont": self.meta_cont_net.state_dict(),
            "cont": self.cont_net.state_dict()
        }

    def load(self, agent_data: Dict):
        meta_cont_data, cont_data = itemgetter("meta_cont", "cont")(agent_data)

        self.meta_cont_net.load_state_dict(meta_cont_data)
        self.meta_cont_net_fixed.load_state_dict(meta_cont_data)

        self.cont_net.load_state_dict(cont_data)
        self.cont_net_fixed.load_state_dict(cont_data)


def _play_into_buffers(
        env: gym.Env,
        agent: HDQNAgent,
        state,
        n_steps=1
):
    s = state
    g = agent._get_epsilon(s, agent.eps1)

def train_h_dqn_agent(
        env_factory: Callable[[int], gym.Env],
        agent: HDQNAgent,
        opt: torch.optim.Optimizer,
        num_episodes: int = 100,
        replay_buffer_size=10**4
):
    # Init all epsilons.
    agent.epsilon1 = 1
    agent.epsilon2 = np.ones(agent.n_goals)

    # Init D1 & D2.
    while len(agent.d1) < replay_buffer_size and len(agent.d2) < replay_buffer_size:
        action = agent.forward(state, record=True)

    for ep in range(num_episodes):
        goal = agent.get_goal()
