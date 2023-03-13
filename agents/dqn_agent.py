import configparser
from operator import itemgetter
from typing import Callable, Dict, Tuple, Any, List

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from agents.utils import StaticLinearDecay, smoothen, can_graph, evaluate, parse_int_list
from agents.replay_buffer import ReplayBuffer
from agents.dqn import DQN, compute_td_loss
from talos import Agent


class DQNAgent(Agent):
    def __init__(self, obs, n_actions, device='cpu'):
        super().__init__("DQN")
        self.epsilon = 1
        self.gamma = 0.99
        self.n_actions = n_actions
        self.device = device

        # Init both networks.
        obs_size = len(obs)
        self.net = DQN(obs_size, n_actions, device=device)
        self.target_net = DQN(obs_size, n_actions, device=device)
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
            is_done.astype('bool'),
            device=self.device,
            dtype=torch.bool,
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
        loss = F.mse_loss(target_qvalues_for_actions, predicted_qvalues_for_actions)

        return loss

        # return compute_td_loss(
        #     states,
        #     actions,
        #     rewards,
        #     next_states,
        #     is_done,
        #     self.gamma,
        #     self.net,
        #     self.target_net
        # )

    def get_action(self, state, extra_state=None) -> Tuple[int, Any]:
        return self.get_optimal_actions(np.array([state]))[0], extra_state
        # return self.get_optimal_actions(state), extra_state

    def get_epsilon_actions(self, states: np.ndarray):
        """Pick actions according to an epsilon greedy strategy."""
        return self._get_actions(states, self.epsilon)

    def get_optimal_actions(self, states: np.ndarray):
        """Pick actions according to a greedy strategy."""
        return self._get_actions(states, 0)

    def _get_actions(self, states: np.ndarray, epsilon: float):
        """Pick actions according to an epsilon greedy strategy."""
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            qvalues = self.forward(states)
        self.net.train()

        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1).cpu()

        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

        # return self.net.get_epsilon(states, epsilon)

    def parameters(self):
        return self.net.parameters()

    def save(self) -> Dict:
        return {
            "data": self.net.state_dict(),
            "layers": self.net.hidden_layers
        }

    def load(self, agent_data: Dict):
        data, layers = itemgetter("data", "layers")(agent_data)
        self.net.set_hidden_layers(layers)
        self.net.load_state_dict(data)
        self.target_net.load_state_dict(data)

    def set_hidden_layers(self, hidden_layers: List[int]):
        self.net.set_hidden_layers(hidden_layers)
        self.target_net.set_hidden_layers(hidden_layers)


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
            s, _ = env.reset()

    return s


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
        hidden_layers: List[int],
        artifacts: Dict,
        opt: torch.optim.Optimizer,
        epsilon_decay: StaticLinearDecay = StaticLinearDecay(1, 0.1, 1 * 10 ** 4),
        max_steps: int = 4 * 10 ** 4,
        timesteps_per_epoch=1,
        batch_size=30,
        update_target_net_freq=100,
        evaluation_freq=1000,
        gather_freq=20,
        replay_buffer_size=10**4,
        max_grad_norm=5000
):
    env = env_factory(0)
    state, _ = env.reset()

    agent.set_hidden_layers(hidden_layers)

    # Create buffer and fill it with experiences.
    buffer = ReplayBuffer(replay_buffer_size)
    agent.epsilon = 1
    for _ in range(100):
        state = _play_into_buffer(env, agent, buffer, state, n_steps=10**2)
        if len(buffer) >= replay_buffer_size:
            break

    state, _ = env.reset()

    if can_graph():
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    else:
        fig, axs = None, None

    mean_reward_history = []
    loss_history = []
    grad_norm_history = []

    artifacts["reward"] = mean_reward_history
    artifacts["loss"] = loss_history
    artifacts["grad_norm"] = grad_norm_history

    for step in trange(0, max_steps):

        agent.epsilon = epsilon_decay.get(step)

        state = _play_into_buffer(env, agent, buffer, state, timesteps_per_epoch)

        (s, a, r, s_dash, is_done) = buffer.sample(batch_size)

        loss = agent.compute_loss(s, a, r, s_dash, is_done)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()

        if step % update_target_net_freq == 0:
            # Load agent weights into target_network
            agent.update_target_net()

        if step % gather_freq == 0:
            # Gather info.
            loss_history.append(loss.data.cpu().numpy())
            grad_norm_history.append(grad_norm.data.cpu().numpy())

        if step % evaluation_freq == 0:
            score = evaluate(env_factory(step), agent, n_episodes=3, max_episode_steps=1000)
            mean_reward_history.append(
                score
            )

            _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history)

            tqdm.write(f"iter: {step: 10.0f} eps: {agent.epsilon: 10.1f}\tscore: {score: 10.2f}\t loss: {loss: 10.2f}")


def _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history):
    if can_graph() is False:
        return

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    axs[0].set_title("Mean Reward")
    axs[1].set_title("Loss")
    axs[2].set_title("Grad Norm")

    axs[0].plot(mean_reward_history)
    axs[1].plot(smoothen(loss_history))
    axs[2].plot(smoothen(grad_norm_history))

    plt.pause(0.05)


def dqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],
        agent: DQNAgent,
        dqn_config: configparser.SectionProxy,
        artifacts: Dict,
):
    train_dqn_agent(
        env_factory=env_factory,
        agent=agent,
        hidden_layers=parse_int_list(dqn_config.get("hidden_layers")),
        artifacts=artifacts,
        opt=torch.optim.NAdam(agent.parameters(), lr=dqn_config.getfloat("learning_rate")),
        epsilon_decay=StaticLinearDecay(
            start_value=dqn_config.getfloat("init_epsilon"),
            final_value=dqn_config.getfloat("final_epsilon"),
            max_steps=dqn_config.getint("decay_steps")
        ),
        max_steps=dqn_config.getint("total_steps"),
        timesteps_per_epoch=dqn_config.getint("timesteps_per_epoch"),
        batch_size=dqn_config.getint("batch_size"),
        update_target_net_freq=dqn_config.getint("refresh_target_network_freq"),
        evaluation_freq=dqn_config.getint("eval_freq"),
        gather_freq=dqn_config.getint("gather_freq"),
        replay_buffer_size=dqn_config.getint("replay_buffer_size")
    )
