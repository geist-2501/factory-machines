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

    def params(self):
        return self.net.parameters()

    def set_hidden_layers(self, hidden_layers: List[int]):
        """
        Set the configuration of the hidden layers.
        Each integer in the list is the number of neurons in each layer.
        Must be done *before* the optimiser is created!
        """
        self.hidden_layers = hidden_layers
        self.net = self._build_net(self.get_layers()).to(self.device)

    def get_epsilon(self, states: torch.Tensor, epsilon: float) -> Union[np.ndarray, int]:
        """
        Pick an action according to an epsilon policy.
        For batches, it returns a numpy array of actions.
        For single instances, it returns an int action.
        """

        if len(states.shape) == 1:
            # Single version.
            n_actions = self.out_size
            should_explore = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
            if should_explore:
                return np.random.choice(n_actions)
            else:
                with torch.no_grad():
                    qvalues = self.forward(states)
                return qvalues.argmax().item()

        elif len(states.shape) == 2:
            # Batch version
            with torch.no_grad():
                qvalues = self.forward(states)

            batch_size, n_actions = qvalues.shape

            random_actions = np.random.choice(n_actions, size=batch_size)
            best_actions = qvalues.argmax(axis=-1).cpu()

            should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
            return np.where(should_explore, random_actions, best_actions)

    def get_all_action_values(self, state: np.ndarray) -> np.ndarray:
        """Get all the scores for each action from the given state."""
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            return np.array(self.forward(state).cpu())

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
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        is_done: np.ndarray,
        gamma: float,
        net: DQN,
        net_fixed: DQN,
        deltas: int = 1
):
    """
    Compute the temporal difference loss for a batch of observations.
    Adapted from the usual TD-error formula into a TDδ-error.
    According to formula $$[(r + gamma^delta * max_{g'} Q(s', g'; theta^-)) - Q(s, g; theta)]^2$$
    """

    actions, is_done, next_states, rewards, states = _tensorise(actions, is_done, net, next_states, rewards, states)
    deltas = torch.tensor(deltas, device=net.device, dtype=torch.int64)

    predicted_qvalues = net(states)
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    with torch.no_grad():
        predicted_next_qvalues = net_fixed(next_states)

    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    target_qvalues_for_actions = rewards + torch.pow(gamma, deltas) * next_state_values
    target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = F.mse_loss(target_qvalues_for_actions, predicted_qvalues_for_actions)

    return loss

def compute_double_td_loss(
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        is_done: np.ndarray,
        gamma: float,
        net: DQN,
        net_fixed: DQN
) -> torch.Tensor:
    actions, is_done, next_states, rewards, states = _tensorise(actions, is_done, net, next_states, rewards, states)

    predicted_qvalues = net(states)
    predicted_qvalues_for_actions = predicted_qvalues.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

    with torch.no_grad():
        qvalues_for_estimates = net(next_states)
        qvalues_for_evaluation = net_fixed(next_states)

    _, actions_from_estimates = qvalues_for_estimates.max(dim=1)  # Take the indices from the max, not the values.
    evaluation_values = qvalues_for_evaluation.gather(dim=1, index=actions_from_estimates.unsqueeze(dim=1)).squeeze()

    target_qvalues_for_actions = rewards + gamma * evaluation_values
    target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = F.mse_loss(target_qvalues_for_actions, predicted_qvalues_for_actions)

    return loss


def _tensorise(actions, is_done, net, next_states, rewards, states):
    """Convert the numpy arrays into tensors. Yes, I could think of a better name but tensorise is funnier."""
    states = torch.tensor(states, device=net.device, dtype=torch.float32)
    actions = torch.tensor(actions, device=net.device, dtype=torch.int64)
    rewards = torch.tensor(rewards, device=net.device, dtype=torch.float32)
    next_states = torch.tensor(next_states, device=net.device, dtype=torch.float32)
    is_done = torch.tensor(
        is_done.astype('bool'),
        device=net.device,
        dtype=torch.bool,
    )

    return actions, is_done, next_states, rewards, states
