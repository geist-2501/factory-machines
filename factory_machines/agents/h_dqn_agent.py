import configparser
import math
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Dict, Callable, Tuple, Any, TypeVar, List, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from tqdm import trange, tqdm

from factory_machines.agents.dqn import DQN, compute_td_loss
from factory_machines.agents.replay_buffer import ReplayBuffer, ReplayBufferWithStats
from factory_machines.agents.timekeeper import KCatchUpTimeKeeper, SerialTimekeeper, TimeKeeper
from factory_machines.agents.utils import can_graph, smoothen, parse_int_list, StaticLinearDecay, \
    SuccessRateWithTimeLimitDecay, BetterReduceLROnPlateau
from talos import Agent, ExtraState, EnvFactory, SaveCallback

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

        self.d1 = ReplayBufferWithStats(10 ** 4, n_goals)
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
        self.d1 = ReplayBufferWithStats(size, self.n_goals)
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
        return [*obs, *self.onehot(goal, self.n_goals)]

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
            # print(f"Picked goal {goal}")
        # Get an action from the controller, incorporating the goal.
        controller_obs = self.to_q1(obs, goal)
        action = self.get_epsilon(controller_obs, epsilon=0, net=self.q1_net)

        return action, goal

    def post_step(self, obs, action, next_obs, extra_state: ExtraState = None) -> ExtraState:
        goal = extra_state
        if self.goal_satisfied(obs, action, next_obs, goal):
            # print(f"Completed goal {goal}")
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
        opt.zero_grad()

        (s, a, r, s_dash, is_done) = buffer.sample(batch_size)

        loss = self.get_loss(s, a, r, s_dash, is_done, net, net_fixed)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()

        return loss, grad_norm

    def save(self) -> Dict:
        return {
            "q2_data": self.q2_net.state_dict(),
            "q2_layers": self.q2_net.hidden_layers,
            "q1_data": self.q1_net.state_dict(),
            "q1_layers": self.q1_net.hidden_layers
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
    def onehot(category, n_categories):
        return [1 if i == category else 0 for i in range(n_categories)]


class HDQNTrainingWrapper:
    def __init__(
            self,
            env_factory: EnvFactory,
            agent: HDQNAgent,
            artifacts: Dict,
            config: configparser.SectionProxy,
            save_callback: SaveCallback
    ) -> None:
        self.env_factory = env_factory
        self.agent = agent
        self.artifacts = artifacts
        self.save_callback = save_callback

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

        self.q1_lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt1, self.pretrain_steps + self.train_steps)
        self.q2_lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt2, self.train_steps)

        # Init all epsilons.
        agent.eps1 = np.ones(agent.n_goals)
        agent.eps2 = 1

        decay_start = config.getfloat("init_epsilon")
        decay_end = config.getfloat("final_epsilon")
        q2_decay_steps = config.getint("q2_decay_steps")
        q1_decay_steps = config.getint("q1_decay_steps")

        self.epsilon1_decay = [SuccessRateWithTimeLimitDecay(decay_start, decay_end, q1_decay_steps, 30, 500)
                               for _ in range(agent.n_goals)]
        self.epsilon2_decay = StaticLinearDecay(decay_start, decay_end, q2_decay_steps)

        self.net_update_freq = config.getint("refresh_target_network_freq")
        self.gather_freq = 50
        self.save_freq = 5000

        self.k_catch_up = config.getint("k_catch_up")

        replay_buffer_size = config.getint("replay_buffer_size")
        self.agent.set_replay_buffer_size(replay_buffer_size)

        self.best_agent_score = (-math.inf, None)
        self.n_save_after_peak = 20
        self.n_since_last_peak = 0

        # Statistics.
        self.axs = _init_graphing()
        self.q1_loss_history = []
        self.q2_loss_history = []
        self.q1_grad_norm_history = []
        self.q2_grad_norm_history = []
        self.q1_mean_reward_history = []
        self.q2_mean_reward_history = []
        self.q2_action_length_history = []
        self.q1_lr_history = []
        self.q2_lr_history = []
        self.picked_goals = np.zeros(agent.n_goals)
        self.n_goal_steps = np.zeros(agent.n_goals)
        self.epsilon_history = np.empty(shape=(0, agent.n_goals + 1))  # +1 for Q2's epsilon.

    def train(self):
        env = self.env_factory(None)
        timekeeper = SerialTimekeeper() if self.k_catch_up is None else KCatchUpTimeKeeper()
        timekeeper.pretrain_mode()

        # Init D1.
        with trange(self.replay_buffer_size, desc="D1 Prewarm") as progress_bar:
            while len(self.agent.d1) < self.replay_buffer_size:
                self.play_episode(env, learn=False, show_progress=False)

                progress_bar.update(len(self.agent.d1) - progress_bar.n)
                progress_bar.refresh()

        # Pretrain Q1.
        with tqdm(total=self.pretrain_steps, desc="Q1 Pretrain") as progress_bar:
            while not self._q1_is_successful() or timekeeper.get_q1_steps() < self.pretrain_steps:
                self.play_episode(env, timekeeper=timekeeper, learn=True)

                if timekeeper.get_q1_steps() > progress_bar.total:
                    progress_bar.total += self.pretrain_steps
                progress_bar.update(timekeeper.get_q1_steps() - progress_bar.n)
                progress_bar.refresh()

        # Init D2.
        with trange(self.replay_buffer_size, desc="D2 Prewarm") as progress_bar:
            while len(self.agent.d2) < self.replay_buffer_size:
                self.play_episode(env, learn=False, show_progress=False)

                progress_bar.update(len(self.agent.d2) - progress_bar.n)
                progress_bar.refresh()

        # Train Q1 & Q2.
        timekeeper.train_mode()
        if type(timekeeper) is KCatchUpTimeKeeper:
            timekeeper.set_k_catch_up(self.k_catch_up)
        with trange(self.train_steps, desc="Q2 Steps") as q2_progress_bar:
            with trange(self.train_steps, desc="Q1 Steps") as q1_progress_bar:
                while timekeeper.get_q2_steps() < self.train_steps or timekeeper.get_q1_steps() < self.train_steps:
                    self.play_episode(env, timekeeper=timekeeper, learn=True, show_progress=True)

                    q2_progress_bar.update(timekeeper.get_q2_steps() - q2_progress_bar.n)
                    q2_progress_bar.refresh()
                    q1_progress_bar.update(timekeeper.get_q1_steps() - q1_progress_bar.n)
                    q1_progress_bar.refresh()

    def _q1_is_successful(self):
        return all([decay.get_success_rate() > 0.98 for decay in self.epsilon1_decay])

    def play_episode(
            self,
            env: gym.Env,
            timekeeper: TimeKeeper = None,
            learn=True,
            show_progress=True
    ):
        if learn:
            assert timekeeper is not None, "Cannot learn without a timekeeper!"

        obs, _ = env.reset()
        meta_r = 0  # Total extrinsic reward gained in the meta-action.
        meta_t = 0  # Total amount of timesteps used in the meta-action.
        meta_obs = None  # Observation at the start of the meta-action.

        goal = None

        # Update Q2 epsilon.
        if learn and timekeeper.should_train_q2():
            self.agent.eps2 = self.epsilon2_decay.get(timekeeper.get_q2_steps())

        for _ in trange(self.episode_max_timesteps, leave=False, disable=not show_progress):

            q1_loss, q1_grad_norm, q2_loss, q2_grad_norm = None, None, None, None

            if goal is None:
                # Pick goal.
                meta_obs = obs
                meta_r = 0
                meta_t = 0

                goal = self.agent.get_epsilon_goal(obs)
                self.picked_goals[goal] += 1

            action = self.agent.get_epsilon_action(obs, goal)
            next_obs, ext_r, done, _, _ = env.step(action)
            meta_r += ext_r
            meta_t += 1

            if timekeeper:
                timekeeper.step_env()

            # Get intrinsic reward for the controller.
            int_r = self.agent.get_intrinsic_reward(obs, action, next_obs, goal)

            self.n_goal_steps[goal] += 1
            self.agent.d1.add(
                self.agent.to_q1(obs, goal),
                action,
                int_r,
                self.agent.to_q1(next_obs, goal),
                False
            )

            if learn:
                # Update Q1 on every step.
                if timekeeper.should_train_q1():
                    timekeeper.step_q1()
                    q1_loss, q1_grad_norm = self.agent.update_net(
                        net=self.agent.q1_net,
                        net_fixed=self.agent.q1_net_fixed,
                        buffer=self.agent.d1,
                        opt=self.opt1,
                        batch_size=self.batch_size,
                        max_grad_norm=self.max_grad_norm
                    )

                    self.q1_lr_sched.step()

                    if timekeeper.get_q1_steps() % self.net_update_freq == 0:
                        self.agent.update_q1_fixed()

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
                    self.agent.eps1[goal] = self.epsilon1_decay[goal].next(timekeeper.get_env_steps(), goal_satisfied, meta_t)

                if learn and timekeeper.should_train_q2():
                    timekeeper.step_q2()
                    q2_loss, q2_grad_norm = self.agent.update_net(
                        net=self.agent.q2_net,
                        net_fixed=self.agent.q2_net_fixed,
                        buffer=self.agent.d2,
                        opt=self.opt2,
                        batch_size=self.batch_size,
                        max_grad_norm=self.max_grad_norm
                    )

                    self.q2_lr_sched.step()

                    if timekeeper.get_q2_steps() % self.net_update_freq == 0:
                        self.agent.update_q2_fixed()

                # Clear the goal, will be re-set next iteration.
                goal = None

            obs = next_obs

            if learn:
                score = self.record_statistics(
                    timekeeper,
                    (q1_loss, q2_loss),
                    (q1_grad_norm, q2_grad_norm),
                )

            if done:
                if timekeeper:
                    timekeeper.step_episode()
                return

        if learn and goal:
            # Episode ended with an incomplete goal - consider it failed.
            self.agent.eps1[goal] = self.epsilon1_decay[goal].next(timekeeper.get_env_steps(), False, meta_t)

        if timekeeper:
            timekeeper.step_episode()

    def record_statistics(
            self,
            timekeeper: TimeKeeper,
            loss,
            grad_norm
    ) -> Optional[float]:
        relevant_steps = timekeeper.get_env_steps()
        if relevant_steps % self.gather_freq == 0:
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

            self.q1_lr_history.append(self.q1_lr_sched.get_last_lr())
            self.q2_lr_history.append(self.q2_lr_sched.get_last_lr())

        if relevant_steps % self.eval_freq == 0:
            # Perform an evaluation.
            return self.evaluate_and_graph(seed=timekeeper.get_q1_steps(), timekeeper=timekeeper)

    @staticmethod
    def evaluate_hdqn(
            env: gym.Env,
            agent: HDQNAgent,
            n_episodes=1,
            max_episode_steps=500,
            only_q1=False,
    ) -> Tuple[float, float, int]:
        extrinsic_rewards = []
        intrinsic_rewards = []
        goals_completed = []
        for _ in range(n_episodes):
            s, _ = env.reset()
            num_goals_completed = 0
            total_extrinsic_reward = 0
            total_intrinsic_reward = 0
            goal = None
            for _ in range(max_episode_steps):
                a, goal = agent.get_action(s, goal, only_q1=only_q1)
                next_s, r, done, _, _ = env.step(a)
                total_extrinsic_reward += r
                total_intrinsic_reward += agent.get_intrinsic_reward(s, a, next_s, goal)
                num_goals_completed += int(agent.goal_satisfied(s, a, next_s, goal))
                goal = agent.post_step(s, a, next_s, goal)
                s = next_s

                if done:
                    break

            goals_completed.append(num_goals_completed)
            extrinsic_rewards.append(total_extrinsic_reward if only_q1 is False else np.nan)
            intrinsic_rewards.append(total_intrinsic_reward / max(1, num_goals_completed))
        return np.mean(extrinsic_rewards).item(), np.mean(intrinsic_rewards).item(), np.mean(goals_completed).item()

    def evaluate_and_graph(self, seed, timekeeper: TimeKeeper) -> float:

        extrinsic_score, intrinsic_score, num_goals_completed = self.evaluate_hdqn(
            self.env_factory(seed),
            self.agent,
            n_episodes=3,
            max_episode_steps=1000,
            only_q1=not timekeeper.should_train_q2()
        )
        self.q2_mean_reward_history.append(extrinsic_score)
        self.q1_mean_reward_history.append(intrinsic_score)

        high_score, _ = self.best_agent_score
        if extrinsic_score != np.nan:
            if extrinsic_score > self.best_agent_score[0]:
                tqdm.write(f"!! New personal best set: {high_score:.2f} -> {extrinsic_score:.2f} !!")
                self.best_agent_score = extrinsic_score, self.agent.save()
                self.n_since_last_peak = 0
            else:
                self.n_since_last_peak += 1
                if self.n_since_last_peak == self.n_save_after_peak:
                    tqdm.write("!! Saving snapshot... !!")
                    self.save_callback(self.agent.save(), self.artifacts, timekeeper.get_env_steps(), f"peak-{extrinsic_score}")

        _update_graphs(
            self.axs,
            (self.q1_mean_reward_history, self.q2_mean_reward_history),
            (self.q1_loss_history, self.q2_loss_history),
            (self.q1_grad_norm_history, self.q2_grad_norm_history),
            self.epsilon_history,
            (self.q1_lr_history, self.q2_lr_history)
        )

        self.artifacts["loss"] = (self.q1_loss_history, self.q2_loss_history)
        self.artifacts["grad_norm"] = (self.q1_loss_history, self.q2_loss_history)
        self.artifacts["mean_reward"] = (self.q1_mean_reward_history, self.q2_mean_reward_history)
        self.artifacts["epsilon"] = self.epsilon_history
        self.artifacts["lr"] = (self.q1_lr_history, self.q2_lr_history)

        success_rates = [f"{eps1.get_success_rate():.2f}" for eps1 in self.epsilon1_decay]

        k_end = timekeeper._k_end if type(timekeeper) is KCatchUpTimeKeeper else "N/A"

        q1_loss = np.nanmean(self.q1_loss_history[-10:]).item()
        q2_loss = np.nanmean(self.q2_loss_history[-10:]).item()
        tqdm.write(f"T[Q1: {timekeeper.get_q1_steps()}, "
                   f"Q2: {timekeeper.get_q2_steps()}, "
                   f"Env: {timekeeper.get_env_steps()}, "
                   f"Eps: {timekeeper.get_episode_steps()}], "
                   f"R[Q1: {intrinsic_score:.2f}, Q2: {extrinsic_score:.2f}], "
                   f"L[Q1: {q1_loss:.3f}, Q2: {q2_loss:.3f}], "
                   f"K[{k_end}], "
                   f"SR[{success_rates}], "
                   f"NG[{num_goals_completed:.2f}], "
                   f"Q2-Len[{np.mean(self.q2_action_length_history[-10:])}], "
                   f"D1[{self.agent.d1.contents}]")

        return extrinsic_score


def hdqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],
        agent: HDQNAgent,
        dqn_config: configparser.SectionProxy,
        artifacts: Dict,
        save_callback
):
    HDQNTrainingWrapper(
        env_factory,
        agent,
        artifacts,
        dqn_config,
        save_callback
    ).train()


def hdqn_graphing_wrapper(
        artifacts: Dict
):
    _update_graphs(
        _init_graphing(),
        mean_reward_history=artifacts["mean_reward"],
        loss_history=artifacts["loss"],
        grad_norm_history=artifacts["grad_norm"],
        epsilon_history=artifacts["epsilon"],
        lr_history=artifacts["lr"]
    )


def _init_graphing():
    if can_graph():
        fig = plt.figure(1, layout="constrained")
        gs = GridSpec(3, 3, figure=fig)
        ax_reward = fig.add_subplot(gs[0, :-1])
        ax_epsilon = fig.add_subplot(gs[0, -1:])
        ax_loss = ax_epsilon.twinx()
        ax_q1_loss = fig.add_subplot(gs[1, :-1])
        ax_q1_grad_norm = fig.add_subplot(gs[1, -1:])
        ax_q2_loss = fig.add_subplot(gs[2, :-1])
        ax_q2_grad_norm = fig.add_subplot(gs[2, -1:])
        axs = (ax_reward, ax_epsilon, ax_loss, ax_q1_loss, ax_q1_grad_norm, ax_q2_loss, ax_q2_grad_norm)
    else:
        fig, axs = None, None

    return axs


def _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history, epsilon_history, lr_history):
    if can_graph() is False:
        return

    plt.figure(1)

    ax_reward, ax_epsilon, ax_lr, ax_q1_loss, ax_q1_grad_norm, ax_q2_loss, ax_q2_grad_norm = axs

    ax_reward.cla()
    ax_epsilon.cla()
    ax_lr.cla()
    ax_q1_loss.cla()
    ax_q1_grad_norm.cla()
    ax_q2_loss.cla()
    ax_q2_grad_norm.cla()

    ax_reward.set_title("Mean Reward")
    ax_epsilon.set_title("Epsilon & LR")
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
        label = "Q1-Out" if i == epsilon_history.shape[1] - 1 else label
        ax_epsilon.plot(epsilon_history[:, i], label=label)
    ax_epsilon.legend()

    ax_lr.plot(lr_history[0], label="Q1 LR", dashes=[1, 1])
    ax_lr.plot(lr_history[1], label="Q2 LR", dashes=[1, 1])
    ax_lr.legend()

    ax_q1_grad_norm.plot(smoothen(grad_norm_history[0]))
    ax_q2_grad_norm.plot(smoothen(grad_norm_history[1]))

    plt.pause(0.05)
