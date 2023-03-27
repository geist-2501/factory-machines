import configparser
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Dict, Callable, Tuple, Any, TypeVar, List

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from tqdm import trange, tqdm

from agents.dqn import DQN, compute_td_loss
from agents.replay_buffer import ReplayBuffer
from agents.timekeeper import TimeKeeper
from agents.utils import can_graph, smoothen, MeteredLinearDecay, parse_int_list, SuccessRateDecay, StaticLinearDecay
from talos import Agent, ExtraState

DictObsType = TypeVar("DictObsType")
FlatObsType = TypeVar("FlatObsType")
ActType = TypeVar("ActType")


class HDQNAgent(Agent, ABC):
    """
    Hierarchical Deep Q-Network agent.
    """

    def __init__(
            self,
            obs: DictObsType,
            n_goals: int,
            n_actions: int,
            device: str = 'cpu'
    ) -> None:
        super().__init__("h-DQN")

        self.device = device
        self.gamma = 0.99

        self.eps1 = np.ones(n_goals)
        self.eps2 = 1

        self.obs_size = len(obs)
        self.n_goals = n_goals
        self.n_actions = n_actions

        self.d1 = ReplayBuffer(10 ** 4)
        self.d2 = ReplayBuffer(10 ** 4)

        self._q2_obs_size = len(self.to_q2(obs))
        self._q1_obs_size = len(self.to_q1(obs, 0))

        # Meta-controller Q network / Q2.
        self.q2_net = DQN(self._q2_obs_size, self.n_goals, device=device)
        self.q2_net_fixed = DQN(self._q2_obs_size, self.n_goals, device=device)
        self.update_q2_fixed()

        # Controller Q network / Q1.
        self.q1_net = DQN(self._q1_obs_size, self.n_actions, device=device)
        self.q1_net_fixed = DQN(self._q1_obs_size, self.n_actions, device=device)
        self.update_q1_fixed()

    def set_replay_buffer_size(self, size):
        self.d1 = ReplayBuffer(size)
        self.d2 = ReplayBuffer(size)

    def get_loss(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            is_done: np.ndarray,
            net: DQN,
            net_fixed: DQN
    ):
        return compute_td_loss(states, actions, rewards, next_states, is_done, self.gamma, net, net_fixed)

    @abstractmethod
    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        raise NotImplementedError

    @abstractmethod
    def goal_satisfied(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        """Process an observation before being fed to Q1."""
        return [*obs, *self._onehot(goal, self.n_goals)]

    @abstractmethod
    def to_q2(self, obs: DictObsType) -> FlatObsType:
        """Process an observation before being fed to Q2."""
        return obs

    def get_action(self, obs: DictObsType, extra_state=None, only_q1=False) -> Tuple[ActType, Any]:
        """Get the optimal action given a state."""

        goal = extra_state  # Treat the extra state we're given as the goal.
        if goal is None:
            if only_q1:
                goal = np.random.choice(self.n_goals)
            else:
                # If no goal is given, get one from the meta-controller.
                meta_controller_obs = self.to_q2(obs)
                goal = self.get_epsilon(meta_controller_obs, epsilon=0, net=self.q2_net)

        # Get an action from the controller, incorporating the goal.
        controller_obs = self.to_q1(obs, goal)
        action = self.get_epsilon(controller_obs, epsilon=0, net=self.q1_net)

        return action, goal

    def post_step(self, obs, action, next_obs, extra_state: ExtraState = None) -> ExtraState:
        goal = extra_state
        if self.goal_satisfied(obs, action, next_obs, goal):
            goal = None
        return goal

    def get_epsilon_action(self, obs: DictObsType, goal: ActType) -> ActType:
        """Get an action from the controller, using the epsilon greedy policy."""
        controller_obs = self.to_q1(obs, goal)
        return self.get_epsilon(controller_obs, self.eps1[goal], self.q1_net)

    def get_epsilon_goal(self, obs: DictObsType) -> ActType:
        """Get a goal from the meta-controller, using the epsilon greedy policy."""
        return self.get_epsilon(self.to_q2(obs), self.eps2, self.q2_net)

    def get_epsilon(self, states: np.ndarray, epsilon: float, net: DQN) -> np.ndarray:
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        return net.get_epsilon(states, epsilon)

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

        loss = self.get_loss(s, a, r, s_dash, is_done, net, net_fixed)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()

        return loss, grad_norm

    def save(self) -> Dict:
        return {
            "q2_data": self.q2_net.state_dict(),
            "q2_layers": self.q2_net.hidden_layers,
            "q1_data": self.q1_net.state_dict(),
            "q1_layers": self.q2_net.hidden_layers
        }

    def load(self, agent_data: Dict):
        q2_layers, q1_layers = itemgetter("q2_layers", "q1_layers")(agent_data)
        self.set_hidden_layers(q1_layers, q2_layers)

        q2_data, q1_data = itemgetter("q2_data", "q1_data")(agent_data)

        self.q2_net.load_state_dict(q2_data)
        self.q2_net_fixed.load_state_dict(q2_data)

        self.q1_net.load_state_dict(q1_data)
        self.q1_net_fixed.load_state_dict(q1_data)

    def update_q1_fixed(self):
        self.q1_net_fixed.load_state_dict(self.q1_net.state_dict())

    def update_q2_fixed(self):
        self.q2_net_fixed.load_state_dict(self.q2_net.state_dict())

    def q1_params(self):
        return self.q1_net.parameters()

    def q2_params(self):
        return self.q2_net.parameters()

    def set_hidden_layers(self, q1_hidden_layers: List[int], q2_hidden_layers: List[int]):
        self.q1_net.set_hidden_layers(q1_hidden_layers)
        self.q1_net_fixed.set_hidden_layers(q1_hidden_layers)
        self.update_q1_fixed()

        self.q2_net.set_hidden_layers(q2_hidden_layers)
        self.q2_net_fixed.set_hidden_layers(q2_hidden_layers)
        self.update_q2_fixed()

    @staticmethod
    def _onehot(category, n_categories):
        return [1 if i == category else 0 for i in range(n_categories)]


class HDQNTrainingWrapper:
    def __init__(
            self,
            env_factory: Callable[[int], gym.Env],
            agent: HDQNAgent,
            artifacts: Dict,
            config: configparser.SectionProxy
    ) -> None:
        self.env_factory = env_factory
        self.agent = agent
        self.artifacts = artifacts

        self.q1_pretrain_thresh = config.getfloat("q1_pretrain_thresh")
        self.pretrain_steps = config.getint("pretrain_steps")
        self.train_steps = config.getint("train_steps")
        self.episode_max_timesteps = config.getint("episode_max_timesteps")
        self.replay_buffer_size = config.getint("replay_buffer_size")
        self.batch_size = config.getint("batch_size")
        self.eval_freq = config.getint("eval_freq")

        self.max_grad_norm = 1000

        # Init the network shapes.
        q1_hidden_layers = parse_int_list(config.get("q1_hidden_layers"))
        q2_hidden_layers = parse_int_list(config.get("q2_hidden_layers"))
        self.agent.set_hidden_layers(q1_hidden_layers, q2_hidden_layers)

        # Init optimisers.
        learning_rate = config.getfloat("learning_rate")
        self.opt1 = torch.optim.NAdam(params=agent.q1_params(), lr=learning_rate)
        self.opt2 = torch.optim.NAdam(params=agent.q2_params(), lr=learning_rate)

        # Init all epsilons.
        agent.eps1 = np.ones(agent.n_goals)
        agent.eps2 = 1

        decay_start = config.getfloat("init_epsilon")
        decay_end = config.getfloat("final_epsilon")
        q2_decay_steps = config.getfloat("q2_decay_steps")
        q1_decay_steps = config.getint("q1_decay_steps")

        self.epsilon1_decay = [SuccessRateDecay(decay_start, decay_end, q1_decay_steps) for _ in range(agent.n_goals)]
        self.epsilon2_decay = StaticLinearDecay(decay_start, decay_end, q2_decay_steps)

        self.net_update_freq = config.getint("refresh_target_network_freq")
        self.gather_freq = 50

        self.k_catch_up = config.getint("k_catch_up")

        # Statistics.
        self.axs = self.init_graphing()
        self.q1_loss_history = []
        self.q2_loss_history = []
        self.q1_grad_norm_history = []
        self.q2_grad_norm_history = []
        self.q1_mean_reward_history = []
        self.q2_mean_reward_history = []
        self.q2_action_length_history = []
        self.epsilon_history = np.empty(shape=(0, agent.n_goals + 1))

    def train(self):
        env = self.env_factory(0)
        timekeeper = TimeKeeper()

        # Init D1.
        with trange(self.replay_buffer_size, desc="D1 Prewarm") as progress_bar:
            while len(self.agent.d1) < self.replay_buffer_size:
                self.play_episode(env, learn=False, q1_only=True, show_progress=False)

                progress_bar.update(len(self.agent.d1) - progress_bar.n)
                progress_bar.refresh()

        # Pretrain Q1.
        timekeeper.pretrain_mode()
        with tqdm(total=self.pretrain_steps, desc="Q1 Pretrain") as progress_bar:
            while not self._q1_is_successful():
                self.play_episode(env, timekeeper=timekeeper, learn=True, q1_only=True)

                if timekeeper.get_steps() > progress_bar.total:
                    progress_bar.total += self.pretrain_steps
                progress_bar.update(timekeeper.get_steps() - progress_bar.n)
                progress_bar.refresh()

        # Init D2.
        with trange(self.replay_buffer_size, desc="D2 Prewarm") as progress_bar:
            while len(self.agent.d2) < self.replay_buffer_size:
                self.play_episode(env, learn=False, show_progress=False)

                progress_bar.update(len(self.agent.d2) - progress_bar.n)
                progress_bar.refresh()

        # Train Q1 & Q2.
        timekeeper.train_mode()
        timekeeper.set_k_catch_up(self.k_catch_up)
        with trange(self.train_steps, desc="Q2 Steps") as q2_progress_bar:
            with trange(self.train_steps, desc="Q1 Steps") as q1_progress_bar:
                while timekeeper.get_meta_steps() < self.train_steps or timekeeper.get_steps() < self.train_steps:

                    self.play_episode(env, timekeeper=timekeeper, learn=True, show_progress=True)

                    q2_progress_bar.update(timekeeper.get_meta_steps() - q2_progress_bar.n)
                    q2_progress_bar.refresh()
                    q1_progress_bar.update(timekeeper.get_steps() - q1_progress_bar.n)
                    q1_progress_bar.refresh()

    def _q1_is_successful(self):
        return all([goal.get_epsilon() < 1 - self.q1_pretrain_thresh for goal in self.epsilon1_decay])

    def play_episode(
            self,
            env: gym.Env,
            timekeeper: TimeKeeper = None,
            learn=True,
            q1_only=False,
            show_progress=True
    ):
        obs, _ = env.reset()
        meta_r = 0
        meta_t = 0
        meta_obs = None
        goal = None

        if learn and not q1_only and self.epsilon2_decay and timekeeper:
            self.agent.eps2 = self.epsilon2_decay.get(timekeeper.get_meta_steps())

        for _ in trange(self.episode_max_timesteps, leave=False, disable=not show_progress):

            q1_loss, q1_grad_norm, q2_loss, q2_grad_norm = None, None, None, None

            meta_t += 1

            if goal is None:
                meta_obs = obs
                meta_r = 0
                meta_t = 0

                if q1_only:
                    goal = np.random.choice(self.agent.n_goals)
                else:
                    # Pick goal.
                    goal = self.agent.get_epsilon_goal(obs)

            action = self.agent.get_epsilon_action(obs, goal)
            next_obs, ext_r, done, _, _ = env.step(action)
            meta_r += ext_r

            # Get intrinsic reward for the controller.
            int_r = self.agent.get_intrinsic_reward(obs, action, next_obs, goal)

            self.agent.d1.add(
                self.agent.to_q1(obs, goal),
                action,
                int_r,
                self.agent.to_q1(next_obs, goal),
                done
            )

            if learn and timekeeper:
                # Update Q1 on every step.
                if timekeeper.should_train_q1():
                    timekeeper.step()
                    q1_loss, q1_grad_norm = self.agent.update_net(
                        net=self.agent.q1_net,
                        net_fixed=self.agent.q1_net_fixed,
                        buffer=self.agent.d1,
                        opt=self.opt1,
                        batch_size=self.batch_size,
                        max_grad_norm=self.max_grad_norm
                    )

                    if timekeeper.get_steps() % self.net_update_freq == 0:
                        self.agent.update_q1_fixed()

                if not q1_only:
                    q2_loss, q2_grad_norm = self.agent.update_net(
                        net=self.agent.q2_net,
                        net_fixed=self.agent.q2_net_fixed,
                        buffer=self.agent.d2,
                        opt=self.opt2,
                        batch_size=self.batch_size,
                        max_grad_norm=self.max_grad_norm
                    )

                    if timekeeper.get_steps() % self.net_update_freq == 0:
                        self.agent.update_q2_fixed()

            goal_satisfied = self.agent.goal_satisfied(obs, action, next_obs, goal)
            if goal_satisfied or done:
                # End of the meta-action.
                self.agent.d2.add(
                    self.agent.to_q2(meta_obs),
                    goal,
                    meta_r,
                    self.agent.to_q2(next_obs),
                    done
                )

                self.q2_action_length_history.append(meta_t)

                if learn and timekeeper.should_train_q1():
                    # Update the epsilon for the completed goal.
                    if self.epsilon1_decay:
                        self.agent.eps1[goal] = self.epsilon1_decay[goal].next(goal_satisfied)

                goal = None

                if learn and timekeeper and not q1_only:
                    timekeeper.meta_step()

            obs = next_obs

            if learn:
                self.record_statistics(
                    timekeeper,
                    (q1_loss, q2_loss),
                    (q1_grad_norm, q2_grad_norm),
                    q1_only=q1_only
                )

            if done:
                return

        if self.epsilon1_decay and goal:
            # Goal failed.
            self.agent.eps1[goal] = self.epsilon1_decay[goal].next(False)

    def record_statistics(
            self,
            timekeeper: TimeKeeper,
            loss,
            grad_norm,
            q1_only=False
    ):
        if timekeeper.get_steps() % self.gather_freq == 0:
            self.epsilon_history = np.append(
                self.epsilon_history,
                [[self.agent.eps2, *self.agent.eps1]],
                axis=0
            )

            q1_loss, q2_loss = loss
            q1_grad_norm, q2_grad_norm = grad_norm

            if q1_loss and q1_grad_norm:
                self.q1_loss_history.append(q1_loss.data.cpu().numpy())
                self.q1_grad_norm_history.append(q1_grad_norm.data.cpu().numpy())

            if q2_loss and q2_grad_norm:
                self.q2_loss_history.append(q2_loss.data.cpu().numpy())
                self.q2_grad_norm_history.append(q2_grad_norm.data.cpu().numpy())

        if timekeeper.get_steps() % self.eval_freq == 0:
            # Perform an evaluation.
            self.evaluate_and_graph(seed=timekeeper.get_steps(), timekeeper=timekeeper, only_q1=q1_only)

    @staticmethod
    def evaluate_hdqn(
            env: gym.Env,
            agent: HDQNAgent,
            n_episodes=1,
            max_episode_steps=10000,
            only_q1=False
    ) -> Tuple[float, float]:
        extrinsic_rewards = []
        intrinsic_rewards = []
        for _ in range(n_episodes):
            s, _ = env.reset()
            total_extrinsic_reward = 0
            total_intrinsic_reward = 0
            goal = None
            for _ in range(max_episode_steps):
                a, goal = agent.get_action(s, goal, only_q1=only_q1)
                next_s, r, done, _, _ = env.step(a)
                total_extrinsic_reward += r
                total_intrinsic_reward += agent.get_intrinsic_reward(s, a, next_s, goal)
                goal = agent.post_step(s, a, next_s, goal)
                s = next_s

                if done:
                    break

            extrinsic_rewards.append(total_extrinsic_reward if only_q1 is False else np.nan)
            intrinsic_rewards.append(total_intrinsic_reward)
        return np.mean(extrinsic_rewards).item(), np.mean(intrinsic_rewards).item()

    @staticmethod
    def init_graphing():
        if can_graph():
            fig = plt.figure(layout="constrained")
            gs = GridSpec(3, 3, figure=fig)
            ax_reward = fig.add_subplot(gs[0, :-1])
            ax_epsilon = fig.add_subplot(gs[0, -1:])
            ax_q1_loss = fig.add_subplot(gs[1, :-1])
            ax_q1_grad_norm = fig.add_subplot(gs[1, -1:])
            ax_q2_loss = fig.add_subplot(gs[2, :-1])
            ax_q2_grad_norm = fig.add_subplot(gs[2, -1:])
            axs = (ax_reward, ax_epsilon, ax_q1_loss, ax_q1_grad_norm, ax_q2_loss, ax_q2_grad_norm)
        else:
            fig, axs = None, None

        return axs

    def evaluate_and_graph(self, seed, timekeeper: TimeKeeper, only_q1=False):

        extrinsic_score, intrinsic_score = self.evaluate_hdqn(
            self.env_factory(seed),
            self.agent,
            n_episodes=3,
            max_episode_steps=500,
            only_q1=only_q1
        )
        self.q2_mean_reward_history.append(extrinsic_score)
        self.q1_mean_reward_history.append(intrinsic_score)

        _update_graphs(
            self.axs,
            (self.q1_mean_reward_history, self.q2_mean_reward_history),
            (self.q1_loss_history, self.q2_loss_history),
            (self.q1_grad_norm_history, self.q2_grad_norm_history),
            self.epsilon_history
        )

        self.artifacts["loss"] = (self.q1_loss_history, self.q2_loss_history)
        self.artifacts["grad_norm"] = (self.q1_loss_history, self.q1_loss_history)
        self.artifacts["mean_reward"] = (self.q1_mean_reward_history, self.q2_mean_reward_history)
        self.artifacts["epsilon"] = self.epsilon_history

        q1_mean_success_rate = np.mean([eps1.get_success_rate() for eps1 in self.epsilon1_decay])

        q1_loss = np.nanmean(self.q1_loss_history[-10:]).item()
        q2_loss = np.nanmean(self.q2_loss_history[-10:]).item()
        tqdm.write(f"T[Q1: {timekeeper.get_steps()}, Q2: {timekeeper.get_meta_steps()}], "
                   f"R[Q1: {intrinsic_score:.2f}, Q2: {extrinsic_score:.2f}], "
                   f"L[Q1: {q1_loss:.3f}, Q2: {q2_loss:.3f}], K[{timekeeper._k_end}], "
                   f"SR[{q1_mean_success_rate:.2f}], Q2-Len[{np.mean(self.q2_action_length_history[-10:])}]")


def hdqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],
        agent: HDQNAgent,
        dqn_config: configparser.SectionProxy,
        artifacts: Dict
):
    HDQNTrainingWrapper(
        env_factory,
        agent,
        artifacts,
        dqn_config
    ).train()


def _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history, epsilon_history):
    if can_graph() is False:
        return

    ax_reward, ax_epsilon, ax_q1_loss, ax_q1_grad_norm, ax_q2_loss, ax_q2_grad_norm = axs

    ax_reward.cla()
    ax_epsilon.cla()
    ax_q1_loss.cla()
    ax_q1_grad_norm.cla()
    ax_q2_loss.cla()
    ax_q2_grad_norm.cla()

    ax_reward.set_title("Mean Reward")
    ax_epsilon.set_title("Epsilon")
    ax_q1_loss.set_title("Loss Q1")
    ax_q1_grad_norm.set_title("Grad Norm Q1")
    ax_q2_loss.set_title("Loss Q2")
    ax_q2_grad_norm.set_title("Grad Norm Q2")

    ax_reward.plot(mean_reward_history[0])
    ax_reward.plot(mean_reward_history[1])

    ax_q1_loss.plot(smoothen(loss_history[0]))
    ax_q2_loss.plot(smoothen(loss_history[1]))

    for i in range(epsilon_history.shape[1]):
        label = "Q2" if i == 0 else f"Q1-{i - 1}"
        ax_epsilon.plot(epsilon_history[:, i], label=label)
    ax_epsilon.legend()

    ax_q1_grad_norm.plot(smoothen(grad_norm_history[0]))
    ax_q2_grad_norm.plot(smoothen(grad_norm_history[1]))

    plt.pause(0.05)
