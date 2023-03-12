from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    """A configurable Deep Q-Network."""
    def __init__(self, state_dim: int, n_actions: int, hidden_layers=None) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = []

        self._layers = [state_dim, *hidden_layers, n_actions]
        self.net = nn.Sequential()
        for i in range(len(self._layers) - 1):
            is_first = i == 0
            if not is_first:
                self.net.append(nn.ReLU())

            in_dim = self._layers[i]
            out_dim = self._layers[i + 1]
            self.net.append(nn.Linear(in_dim, out_dim))

        self.net.modules()

    def forward(self, states):
        return self.net(states)

    def get_layers(self):
        return self._layers

    def get_epsilon(self, states: torch.Tensor, epsilon: float) -> np.ndarray:

        with torch.no_grad():
            qvalues = self.forward(states)

        if len(states.shape) == 1:
            # Single version.
            n_actions = qvalues.shape[0]
            should_explore = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
            if should_explore:
                return np.random.choice(n_actions)
            else:
                return qvalues.argmax().item()

        elif len(states.shape) == 2:
            # Batch version
            batch_size, n_actions = qvalues.shape

            random_actions = np.random.choice(n_actions, size=batch_size)
            best_actions = qvalues.argmax(axis=-1).cpu()

            should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
            return np.where(should_explore, random_actions, best_actions)


def compute_td_loss(
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        is_done: torch.Tensor,
        gamma: float,
        net: DQN,
        net_fixed: DQN
):
    """
    Compute the temporal difference loss for a batch of observations.
    According to formula $$[(r + gamma max_{g'} Q(s', g'; theta^-)) - Q(s, g; theta)]^2$$
    """

    predicted_qvalues = net(states)

    with torch.no_grad():
        predicted_next_qvalues = net_fixed(next_states)

    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    target_qvalues_for_actions = rewards + gamma * next_state_values
    target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = F.mse_loss(target_qvalues_for_actions, predicted_qvalues_for_actions)

    return loss
