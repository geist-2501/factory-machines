from typing import Dict, Callable, Tuple, Any
from operator import itemgetter

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange

from factory_machines.dqn_agent import DQN
from factory_machines.replay_buffer import ReplayBuffer
from factory_machines.utils import can_graph, evaluate, smoothen
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

        self.eps1 = np.ones(n_goals)
        self.eps2 = 1

        self.n_goals = n_goals

        self.d1 = ReplayBuffer(10**4)
        self.d2 = ReplayBuffer(10**4)

        self.meta_cont_net = DQN(obs_size, n_goals).to(device)  # Meta-controller net / Q2.
        self.meta_cont_net_fixed = DQN(obs_size, n_goals).to(device)  # Meta-controller fixed net.
        self.cont_net = DQN(n_goals, n_actions).to(device)  # Controller net / Q1.
        self.cont_net_fixed = DQN(n_goals, n_actions).to(device)  # Controller fixed net.

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

    def get_intrinsic_reward(self, obs: np.ndarray,  action: int, next_obs: np.ndarray, goal: int) -> float:
        return 0.1  # TODO implement.

    def goal_satisfied(self, obs: np.ndarray, goal: int) -> bool:
        # TODO implement.
        return np.random.choice([True, False])

    def get_action(self, state: np.ndarray, extra_state=None) -> Tuple[int, Any]:
        """Get the optimal action given a state."""
        goal = extra_state
        meta_controller_obs = np.expand_dims(state, axis=0)
        if goal is None:
            goal = self.get_epsilon(meta_controller_obs, 0, self.meta_cont_net)[0]

        controller_obs = np.append(meta_controller_obs, goal)
        action = self.get_epsilon(controller_obs, epsilon=0, net=self.cont_net)[0]

        return action, goal

    def get_epsilon_action(self, obs: np.ndarray, goal: int):
        """Get an action from the controller, using the epsilon greedy policy."""
        controller_obs = np.concatenate(obs, goal)
        return self.get_epsilon(controller_obs, self.eps1[goal], self.cont_net)

    def get_epsilon_goal(self, obs: np.ndarray):
        """Get a goal from the meta-controller, using the epsilon greedy policy."""
        return self.get_epsilon(obs, self.eps2, self.meta_cont_net)

    def update_net(
            self,
            net: DQN,
            net_fixed: DQN,
            buffer: ReplayBuffer,
            batch_size,
            opt,
            max_grad_norm
    ):

        (s, a, r, s_dash, is_done) = buffer.sample(batch_size)

        loss = self.compute_td_loss(s, a, r, s_dash, is_done, net, net_fixed)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()

        # TODO I'm unsure about this. In regular DQN the fixed net is updated every K-steps.
        net_fixed.load_state_dict(net.state_dict())

        return loss, grad_norm

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


def _play_episode(
        env: gym.Env,
        agent: HDQNAgent,
        opt: torch.optim.Optimizer,
        batch_size,
        max_grad_norm,
        max_timesteps=1000,
        learn=False,
        gather_freq=20
):
    q1_loss_history = []
    q2_loss_history = []
    q1_grad_norm_history = []
    q2_grad_norm_history = []

    s, _ = env.reset()
    g = agent.get_epsilon_goal(s)
    meta_r = 0
    meta_s = None
    for step in range(max_timesteps):
        if g is None:
            # Start step for the meta controller.
            g = agent.get_epsilon_goal(s)
            # Take note of the start state, so we can store it in the buffer later.
            meta_s = s
            meta_r = 0

        # get action from controller.
        a = agent.get_epsilon_action(s, g)

        next_s, ext_r, done, _, _ = env.step(a)

        int_r = agent.get_intrinsic_reward(s, a, next_s, g)
        meta_r += ext_r

        agent.d1.add([*s, g], a, int_r, next_s, done)

        if learn:
            # Update nets.
            q1_loss, q1_grad_norm = agent.update_net(
                net=agent.cont_net,
                net_fixed=agent.cont_net_fixed,
                buffer=agent.d1,
                opt=opt,
                batch_size=batch_size,
                max_grad_norm=max_grad_norm
            )
            q2_loss, q2_grad_norm = agent.update_net(
                net=agent.meta_cont_net,
                net_fixed=agent.meta_cont_net_fixed,
                buffer=agent.d2,
                opt=opt,
                batch_size=batch_size,
                max_grad_norm=max_grad_norm
            )

            if step % gather_freq == 0:
                q1_loss_history.append(q1_loss.data.cpu().numpy())
                q2_loss_history.append(q2_loss.data.cpu().numpy())
                q1_grad_norm_history.append(q1_grad_norm.data.cpu().numpy())
                q2_grad_norm_history.append(q2_grad_norm.data.cpu().numpy())

        if agent.goal_satisfied(s, g):
            # End of the meta-action.
            agent.d2.add(meta_s, g, ext_r, next_s, done)
            g = None

        s = next_s

        if done:
            break

    return np.array([q1_loss_history, q2_loss_history]), \
        np.array([q1_grad_norm_history, q2_grad_norm_history])


def train_h_dqn_agent(
        env_factory: Callable[[int], gym.Env],
        agent: HDQNAgent,
        opt: torch.optim.Optimizer,
        num_episodes: int = 100,
        max_timesteps=1000,
        replay_buffer_size=10**4,
        batch_size=32,
        max_grad_norm=1000,
        eval_freq=10
):
    """Train the hDQN agent following the algorithm outlined by Kulkarni et al. 2016."""
    # Init graphing.
    if can_graph():
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    else:
        fig, axs = None, None

    loss_history = np.array([])
    grad_norm_history = np.array([])
    mean_reward_history = []

    # Init all epsilons.
    agent.epsilon1 = np.ones(agent.n_goals)
    agent.epsilon2 = 1

    env = env_factory(0)
    s, _ = env.reset()

    # Init D1 & D2.
    while len(agent.d1) < replay_buffer_size and len(agent.d2) < replay_buffer_size:
        _play_episode(
            env=env,
            agent=agent,
            opt=opt,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            max_timesteps=replay_buffer_size,
            learn=False
        )

    for ep in trange(0, num_episodes):
        ep_loss_history, ep_grad_norm_history = _play_episode(
            env=env,
            agent=agent,
            opt=opt,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            max_timesteps=max_timesteps,
            learn=True
        )

        np.append(loss_history, ep_loss_history, axis=1)
        np.append(grad_norm_history, ep_grad_norm_history, axis=1)

        # TODO Anneal epsilons.

        if ep % eval_freq == 0:
            score = evaluate(env_factory(ep), agent, n_episodes=3, max_episode_steps=1000)
            mean_reward_history.append(
                score
            )

            _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history)


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
