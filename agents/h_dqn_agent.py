import configparser
from typing import Dict, Callable, Tuple, Any, TypeVar, List, Optional
from operator import itemgetter
from abc import ABC, abstractmethod

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from agents.dqn_agent import DQN
from agents.replay_buffer import ReplayBuffer
from agents.utils import can_graph, evaluate, smoothen, StaticLinearDecay, MeteredLinearDecay
from talos import Agent


DictObsType = TypeVar("DictObsType")
FlatObsType = TypeVar("FlatObsType")
ActType = TypeVar("ActType")


class HDQNAgent(Agent, ABC):

    def __init__(
            self,
            obs: DictObsType,
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

        q2_obs_size = len(self.to_q2(obs))
        q1_obs_size = len(self.to_q1(obs, 0))

        # Meta-controller Q network / Q2.
        self.meta_cont_net = DQN(q2_obs_size, n_goals).to(device)
        self.meta_cont_net_fixed = DQN(q2_obs_size, n_goals).to(device)

        # Controller Q network / Q1.
        self.cont_net = DQN(q1_obs_size, n_actions).to(device)
        self.cont_net_fixed = DQN(q1_obs_size, n_actions).to(device)

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
        with torch.no_grad():  # TODO add this change to DQN too!
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

    @abstractmethod
    def get_intrinsic_reward(self, obs: DictObsType,  action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        raise NotImplementedError

    @abstractmethod
    def goal_satisfied(self, obs: DictObsType, goal: ActType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        """Process an observation before being fed to Q1."""
        return [*obs, *self._onehot(goal, self.n_goals)]

    @abstractmethod
    def to_q2(self, obs: DictObsType) -> FlatObsType:
        """Process an observation before being fed to Q2."""
        raise NotImplementedError

    def get_action(self, obs: DictObsType, extra_state=None) -> Tuple[ActType, Any]:
        """Get the optimal action given a state."""
        goal = extra_state  # Treat the extra state we're given as the goal.
        if goal is None:
            # If no goal is given, get one from the meta-controller.
            meta_controller_obs = self.to_q2(obs)
            goal = self.get_epsilon(meta_controller_obs, epsilon=0, net=self.meta_cont_net)

        # Get an action from the controller, incorporating the goal.
        controller_obs = self.to_q1(obs, goal)
        action = self.get_epsilon(controller_obs, epsilon=0, net=self.cont_net)

        return action, goal

    def get_epsilon_action(self, obs: DictObsType, goal: ActType) -> ActType:
        """Get an action from the controller, using the epsilon greedy policy."""
        controller_obs = self.to_q1(obs, goal)
        return self.get_epsilon(controller_obs, self.eps1[goal], self.cont_net)

    def get_epsilon_goal(self, obs: DictObsType) -> ActType:
        """Get a goal from the meta-controller, using the epsilon greedy policy."""
        return self.get_epsilon(self.to_q2(obs), self.eps2, self.meta_cont_net)

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

        # TODO I'm unsure about this. In regular DQN the fixed net is updated every K-steps.
        net_fixed.load_state_dict(net.state_dict())

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()

        return loss, grad_norm

    def get_epsilon(self, states: np.ndarray, epsilon: float, net: DQN) -> np.ndarray:
        states = torch.tensor(states, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            qvalues = net(states)

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

    @staticmethod
    def _onehot(category, n_categories):
        return [1 if i == category else 0 for i in range(n_categories)]

    def q1_params(self):
        return self.cont_net.parameters()

    def q2_params(self):
        return self.meta_cont_net.parameters()


def _play_episode(
        env: gym.Env,
        agent: HDQNAgent,
        opt1: torch.optim.Optimizer,
        opt2: torch.optim.Optimizer,
        batch_size,
        max_grad_norm,
        max_timesteps=1000,
        learn=False,
        epsilon1_decay: List[MeteredLinearDecay] = None,
        epsilon2_decay: MeteredLinearDecay = None,
        gather_freq=20,
        meta_action_timelimit: Optional[int] = 100,
        show_progress=False
):
    q1_loss_history = []
    q2_loss_history = []
    q1_grad_norm_history = []
    q2_grad_norm_history = []

    # Update the meta-controller's epsilon at the start of each episode.
    if epsilon2_decay:
        agent.eps2 = epsilon2_decay.next()

    s, _ = env.reset()
    g = agent.get_epsilon_goal(s)
    meta_r = 0
    meta_s = s
    meta_t = 0  # Timesteps taken for the meta-action.

    step_iter = trange(max_timesteps, leave=False) if show_progress else range(max_timesteps)
    for step in step_iter:

        meta_t += 1

        if g is None:
            # Start step for the meta controller.
            g = agent.get_epsilon_goal(s)
            # Take note of the start state, so we can store it in the buffer later.
            meta_s = s
            meta_r = 0
            meta_t = 0

        # get action from controller.
        a = agent.get_epsilon_action(s, g)

        # Step the env and get extrinsic reward for the meta-controller.
        next_s, ext_r, done, _, _ = env.step(a)
        meta_r += ext_r

        # Get intrinsic reward for the controller.
        int_r = agent.get_intrinsic_reward(s, a, next_s, g)

        agent.d1.add(agent.to_q1(s, g), a, int_r, agent.to_q1(next_s, g), done)

        if learn:
            # Update nets.
            q1_loss, q1_grad_norm = agent.update_net(
                net=agent.cont_net,
                net_fixed=agent.cont_net_fixed,
                buffer=agent.d1,
                opt=opt1,
                batch_size=batch_size,
                max_grad_norm=max_grad_norm
            )
            q2_loss, q2_grad_norm = agent.update_net(
                net=agent.meta_cont_net,
                net_fixed=agent.meta_cont_net_fixed,
                buffer=agent.d2,
                opt=opt2,
                batch_size=batch_size,
                max_grad_norm=max_grad_norm
            )

            if step % gather_freq == 0:
                q1_loss_history.append(q1_loss.data.cpu().numpy())
                q2_loss_history.append(q2_loss.data.cpu().numpy())
                q1_grad_norm_history.append(q1_grad_norm.data.cpu().numpy())
                q2_grad_norm_history.append(q2_grad_norm.data.cpu().numpy())

        timelimit_exceeded = meta_t > meta_action_timelimit if meta_action_timelimit else False
        if agent.goal_satisfied(s, g) or timelimit_exceeded:
            # End of the meta-action.
            agent.d2.add(agent.to_q2(meta_s), g, ext_r, agent.to_q2(next_s), done)

            # Update the epsilon for the completed goal.
            if epsilon1_decay:
                agent.eps1[g] = epsilon1_decay[g].next()

            g = None

        s = next_s

        if done:
            break

    return np.array([q1_loss_history, q2_loss_history]), \
        np.array([q1_grad_norm_history, q2_grad_norm_history])


def train_h_dqn_agent(
        env_factory: Callable[[int], gym.Env],
        agent: HDQNAgent,
        opt1: torch.optim.Optimizer,
        opt2: torch.optim.Optimizer,
        num_episodes: int = 100,
        max_timesteps=1000,
        replay_buffer_size=10**4,
        batch_size=32,
        max_grad_norm=1000,
        eval_freq=5,
        decay_start=1,
        decay_end=0.1,
        decay_steps=10**4
):
    """Train the hDQN agent following the algorithm outlined by Kulkarni et al. 2016."""
    # Init graphing.
    if can_graph():
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = np.array(axs).flatten()
        axs1twin = axs[1].twinx()
        axs2twin = axs[2].twinx()
        axs = np.insert(axs, 2, axs1twin)
        axs = np.insert(axs, 4, axs2twin)

    else:
        fig, axs = None, None

    loss_history = np.empty(shape=(2, 0))
    grad_norm_history = np.empty(shape=(2, 0))
    mean_reward_history = []
    epsilon_history = np.empty(shape=(0, agent.n_goals + 1))

    # Init all epsilons.
    agent.epsilon1 = np.ones(agent.n_goals)
    agent.epsilon2 = 1

    epsilon1_decay = [MeteredLinearDecay(decay_start, decay_end, decay_steps) for _ in range(agent.n_goals)]
    epsilon2_decay = MeteredLinearDecay(decay_start, decay_end, decay_steps)

    env = env_factory(0)
    s, _ = env.reset()

    # Init D1 & D2.
    replay_bar = tqdm(range(replay_buffer_size))
    replay_bar.set_description("Warming buffer")
    while len(agent.d1) < replay_buffer_size or len(agent.d2) < replay_buffer_size:
        _play_episode(
            env=env,
            agent=agent,
            opt1=opt1,
            opt2=opt2,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            max_timesteps=replay_buffer_size,
            learn=False
        )

        replay_bar.update(len(agent.d2) - replay_bar.n)
        replay_bar.refresh()

    replay_bar.close()

    for ep in trange(0, num_episodes):
        ep_loss_history, ep_grad_norm_history = _play_episode(
            env=env,
            agent=agent,
            opt1=opt1,
            opt2=opt2,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            max_timesteps=max_timesteps,
            learn=True,
            epsilon1_decay=epsilon1_decay,
            epsilon2_decay=epsilon2_decay,
            show_progress=True,
            meta_action_timelimit=None
        )

        epsilon_history = np.append(
            epsilon_history,
            [[agent.eps2, *agent.eps1]],
            axis=0
        )

        loss_history = np.append(loss_history, ep_loss_history, axis=1)
        grad_norm_history = np.append(grad_norm_history, ep_grad_norm_history, axis=1)

        if ep % eval_freq == 0:
            score = evaluate(env_factory(ep), agent, n_episodes=3, max_episode_steps=1000)
            mean_reward_history.append(
                score
            )

            _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history, epsilon_history)


def hdqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],
        agent: HDQNAgent,
        dqn_config: configparser.SectionProxy
):
    train_h_dqn_agent(
        env_factory=env_factory,
        agent=agent,
        opt1=torch.optim.NAdam(agent.q1_params(), lr=dqn_config.getfloat("learning_rate")),
        opt2=torch.optim.NAdam(agent.q2_params(), lr=dqn_config.getfloat("learning_rate")),
        num_episodes=dqn_config.getint("num_episodes"),
        max_timesteps=dqn_config.getint("total_steps"),
        replay_buffer_size=dqn_config.getint("replay_buffer_size"),
        batch_size=dqn_config.getint("batch_size"),
        decay_start=dqn_config.getfloat("init_epsilon"),
        decay_end=dqn_config.getfloat("final_epsilon"),
        decay_steps=dqn_config.getfloat("decay_steps")
    )


def _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history, epsilon_history):
    if can_graph() is False:
        return

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    axs[3].cla()
    axs[4].cla()
    axs[5].cla()

    axs[0].set_title("Mean Reward")
    axs[1].set_title("Loss")
    axs[3].set_title("Grad Norm")
    axs[5].set_title("Epsilon")

    axs[0].plot(mean_reward_history)

    axs[1].plot(smoothen(loss_history[0]), color="red")
    axs[1].set_ylabel("Q1", color="red")
    axs[2].plot(smoothen(loss_history[1]), color="blue")
    axs[2].set_ylabel("Q2", color="blue")

    axs[3].plot(smoothen(grad_norm_history[0]), color="red")
    axs[3].set_ylabel("Q1", color="red")
    axs[4].plot(smoothen(grad_norm_history[1]), color="blue")
    axs[4].set_ylabel("Q2", color="blue")

    for i in range(epsilon_history.shape[1]):
        label = "Q2" if i == 0 else f"Q1-{i}"
        axs[5].plot(epsilon_history[:, i], label=label)

    plt.tight_layout()
    plt.pause(0.05)
