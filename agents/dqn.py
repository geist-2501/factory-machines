from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    """A configurable Deep Q-Network."""
    def __init__(self, state_dim: int, n_actions: int, hidden_layers=None, device='cpu') -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = []

        self.device = device

        self.in_size = state_dim
        self.out_size = n_actions
        self.hidden_layers = hidden_layers
        self.net = self._build_net(self.get_layers()).to(self.device)

    def forward(self, states):
        return self.net(states)

    def get_layers(self):
        return [self.in_size, *self.hidden_layers, self.out_size]

    def set_hidden_layers(self, hidden_layers: List[int]):
        """
        Set the configuration of the hidden layers.
        Each integer in the list is the number of neurons in each layer.
        """
        self.hidden_layers = hidden_layers
        self.net = self._build_net(self.get_layers()).to(self.device)

    def get_epsilon(self, states: torch.Tensor, epsilon: float) -> Union[np.ndarray, int]:

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

    @staticmethod
    def _build_net(layers: List[int]) -> nn.Sequential:
        net = nn.Sequential()
        for i in range(len(layers) - 1):
            is_first = i == 0
            if not is_first:
                net.append(nn.ReLU())

            in_dim = layers[i]
            out_dim = layers[i + 1]
            net.append(nn.Linear(in_dim, out_dim))

        return net


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
