import configparser
from typing import Dict, Callable, Tuple, Any, TypeVar, List, Optional
from operator import itemgetter
from abc import ABC, abstractmethod

import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gym import Env
from tqdm import trange, tqdm

from agents.dqn import DQN, compute_td_loss
from agents.replay_buffer import ReplayBuffer
from agents.utils import can_graph, evaluate, smoothen, MeteredLinearDecay, parse_int_list
from talos import Agent

DictObsType = TypeVar("DictObsType")
FlatObsType = TypeVar("FlatObsType")
ActType = TypeVar("ActType")


class TimeKeeper:
    def __init__(self) -> None:
        super().__init__()
        self.steps = 0
        self.meta_steps = 0


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

        # Controller Q network / Q1.
        self.q1_net = DQN(self._q1_obs_size, self.n_actions, device=device)
        self.q1_net_fixed = DQN(self._q1_obs_size, self.n_actions, device=device)

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

    def get_action(self, obs: DictObsType, extra_state=None) -> Tuple[ActType, Any]:
        """Get the optimal action given a state."""
        goal = extra_state  # Treat the extra state we're given as the goal.
        if goal is None:
            # If no goal is given, get one from the meta-controller.
            meta_controller_obs = self.to_q2(obs)
            goal = self.get_epsilon(meta_controller_obs, epsilon=0, net=self.q2_net)

        # Get an action from the controller, incorporating the goal.
        controller_obs = self.to_q1(obs, goal)
        action = self.get_epsilon(controller_obs, epsilon=0, net=self.q1_net)

        return action, goal

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

        net_fixed.load_state_dict(net.state_dict())

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

        self.q2_net.set_hidden_layers(q2_hidden_layers)
        self.q2_net_fixed.set_hidden_layers(q2_hidden_layers)

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

        self.num_episodes = config.getint("num_episodes")
        self.max_timesteps = config.getint("total_steps")
        self.replay_buffer_size = config.getint("replay_buffer_size")
        self.batch_size = config.getint("batch_size")
        self.eval_freq = config.getint("eval_freq"),

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
        decay_steps = config.getfloat("decay_steps")
        self.epsilon1_decay = [MeteredLinearDecay(decay_start, decay_end, decay_steps // agent.n_goals)
                               for _ in range(agent.n_goals)]
        self.epsilon2_decay = MeteredLinearDecay(decay_start, decay_end, decay_steps)

        self.net_update_freq = 100

        self.axs = self.init_graphing()
        self.loss_history = np.empty(shape=(2, 0))
        self.grad_norm_history = np.empty(shape=(2, 0))
        self.mean_reward_history = []
        self.epsilon_history = np.empty(shape=(0, agent.n_goals + 1))

    def train(self):
        env = self.env_factory(0)
        timekeeper = TimeKeeper()
        obs, _ = env.reset()

        # Init D1.
        with trange(self.replay_buffer_size) as progress_bar:
            progress_bar.set_description("D1 Prewarm")
            while len(self.agent.d1) < self.replay_buffer_size:
                random_goal = np.random.choice(self.agent.n_goals)
                _, _, done = self.play_q1(env, obs, random_goal, learn_q1=False)
                if done:
                    obs, _ = env.reset()

                progress_bar.update(len(self.agent.d1) - progress_bar.n)
                progress_bar.refresh()

        # Pretrain Q1.
        obs, _ = env.reset()
        with trange(self.num_episodes) as progress_bar:
            progress_bar.set_description("D1 Training")
            for _ in progress_bar:
                random_goal = np.random.choice(self.agent.n_goals)
                _, _, done = self.play_q1(env, obs, random_goal, learn_q1=True, timekeeper=timekeeper)
                if done:
                    obs, _ = env.reset()

        # Init D2.
        obs, _ = env.reset()
        with trange(self.replay_buffer_size) as progress_bar:
            progress_bar.set_description("D2 Prewarm")
            while len(self.agent.d2) < self.replay_buffer_size:
                self.play_q2(env, learn_q2=False, learn_q1=False)

                progress_bar.update(len(self.agent.d2) - progress_bar.n)
                progress_bar.refresh()

        # Train Q1 & Q2.
        with trange(self.num_episodes) as progress_bar:
            progress_bar.set_description("Episodes")
            for step in progress_bar:
                obs, _ = env.reset()
                self.play_q2(env, timekeeper=timekeeper)

                epsilon_history = np.append(
                    epsilon_history,
                    [[self.agent.eps2, *self.agent.eps1]],
                    axis=0
                )
                # loss_history = np.append(loss_history, ep_loss_history, axis=1)
                # grad_norm_history = np.append(grad_norm_history, ep_grad_norm_history, axis=1)

                if step % self.eval_freq == 0:
                    # Perform an evaluation.
                    self.evaluate_and_graph(seed=step, timekeeper=timekeeper)

    def play_q2(
            self,
            env: gym.Env,
            learn_q1=True,
            learn_q2=True,
            timekeeper: TimeKeeper = None
    ):
        # Update the meta-controller's epsilon at the start of each episode.
        if learn_q2:
            self.agent.eps2 = self.epsilon2_decay.next()

        obs, _ = env.reset()
        for _ in trange(self.max_timesteps, leave=False):
            # Get goal.
            goal = self.agent.get_epsilon_goal(obs)

            # Take meta-step.
            next_obs, reward, is_done = self.play_q1(env, obs, goal, learn_q1)

            # Add to replay buffer.
            self.agent.d2.add(
                self.agent.to_q2(obs),
                goal,
                reward,
                self.agent.to_q2(next_obs),
                is_done
            )

            if learn_q2:
                q2_loss, q2_grad_norm = self.agent.update_net(
                    net=self.agent.q2_net,
                    net_fixed=self.agent.q2_net_fixed,
                    buffer=self.agent.d2,
                    opt=self.opt2,
                    batch_size=self.batch_size,
                    max_grad_norm=self.max_grad_norm
                )

            if timekeeper:
                timekeeper.meta_steps += 1

                if timekeeper.meta_steps % self.net_update_freq == 0:
                    self.agent.update_q2_fixed()

            if is_done:
                break

    def play_q1(
            self,
            env: gym.Env,
            obs: DictObsType,
            goal: ActType,
            learn_q1=True,
            timekeeper: TimeKeeper = None
    ) -> Tuple[DictObsType, int, bool]:
        ext_r = 0
        for step in range(self.max_timesteps):

            action = self.agent.get_epsilon_action(obs, goal)

            # Step the env and get extrinsic reward for the meta-controller.
            next_obs, r, done, _, _ = env.step(action)
            ext_r += r

            # Get intrinsic reward for the controller.
            int_r = self.agent.get_intrinsic_reward(obs, action, next_obs, goal)

            self.agent.d1.add(self.agent.to_q1(obs, goal), action, int_r, self.agent.to_q1(next_obs, goal), done)

            if learn_q1:
                q1_loss, q1_grad_norm = self.agent.update_net(
                    net=self.agent.q1_net,
                    net_fixed=self.agent.q1_net_fixed,
                    buffer=self.agent.d1,
                    opt=self.opt1,
                    batch_size=self.batch_size,
                    max_grad_norm=self.max_grad_norm
                )

            if timekeeper:
                timekeeper.steps += 1

                if timekeeper.steps % self.net_update_freq == 0:
                    self.agent.update_q1_fixed()

            if self.agent.goal_satisfied(obs, action, next_obs, goal) or done:
                # Update the epsilon for the completed goal.
                if learn_q1:
                    self.agent.eps1[goal] = self.epsilon1_decay[goal].next()
                return obs, ext_r, done

        return obs, ext_r, False

    @staticmethod
    def init_graphing():
        if can_graph():
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            axs = np.array(axs).flatten()
            axs1twin = axs[1].twinx()
            axs2twin = axs[2].twinx()
            axs = np.insert(axs, 2, axs1twin)
            axs = np.insert(axs, 4, axs2twin)
        else:
            fig, axs = None, None

        return axs

    def evaluate_and_graph(self, seed, timekeeper: TimeKeeper):

        score = evaluate(self.env_factory(seed), self.agent, n_episodes=3, max_episode_steps=500)
        self.mean_reward_history.append(score)

        _update_graphs(
            self.axs,
            self.mean_reward_history,
            self.loss_history,
            self.grad_norm_history,
            self.epsilon_history
        )

        self.artifacts["loss"] = self.loss_history
        self.artifacts["grad_norm"] = self.grad_norm_history
        self.artifacts["mean_reward"] = self.mean_reward_history
        self.artifacts["epsilon"] = self.epsilon_history

        tqdm.write(f"Timesteps: {timekeeper.steps}, meta steps: {timekeeper.meta_steps}")


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
        show_progress=False,
        timekeeper: Optional[TimeKeeper] = None,
        net_update_freq=20
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

        # Get action from controller.
        a = agent.get_epsilon_action(s, g)

        # Step the env and get extrinsic reward for the meta-controller.
        next_s, ext_r, done, _, _ = env.step(a)
        meta_r += ext_r

        # Get intrinsic reward for the controller.
        int_r = agent.get_intrinsic_reward(s, a, next_s, g)

        agent.d1.add(agent.to_q1(s, g), a, int_r, agent.to_q1(next_s, g), done)

        if timekeeper:
            timekeeper.steps += 1

        if learn:
            # Update nets.
            q1_loss, q1_grad_norm = agent.update_net(
                net=agent.q1_net,
                net_fixed=agent.q1_net_fixed,
                buffer=agent.d1,
                opt=opt1,
                batch_size=batch_size,
                max_grad_norm=max_grad_norm
            )
            q2_loss, q2_grad_norm = agent.update_net(
                net=agent.q2_net,
                net_fixed=agent.q2_net_fixed,
                buffer=agent.d2,
                opt=opt2,
                batch_size=batch_size,
                max_grad_norm=max_grad_norm
            )

            if timekeeper:
                if timekeeper.steps % net_update_freq == 0:
                    agent.update_q1_fixed()

                if timekeeper.meta_steps % net_update_freq == 0:
                    agent.update_q2_fixed()

            if step % gather_freq == 0:
                q1_loss_history.append(q1_loss.data.cpu().numpy())
                q2_loss_history.append(q2_loss.data.cpu().numpy())
                q1_grad_norm_history.append(q1_grad_norm.data.cpu().numpy())
                q2_grad_norm_history.append(q2_grad_norm.data.cpu().numpy())

        timelimit_exceeded = meta_t > meta_action_timelimit if meta_action_timelimit else False
        if agent.goal_satisfied(s, a, next_s, g) or timelimit_exceeded:
            # End of the meta-action.
            agent.d2.add(agent.to_q2(meta_s), g, ext_r, agent.to_q2(next_s), done)

            # Update the epsilon for the completed goal.
            if epsilon1_decay:
                agent.eps1[g] = epsilon1_decay[g].next()

            if timekeeper:
                timekeeper.meta_steps += 1

            g = None

        s = next_s

        if done:
            break

    return np.array([q1_loss_history, q2_loss_history]), \
        np.array([q1_grad_norm_history, q2_grad_norm_history])


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
        label = "Q2" if i == 0 else f"Q1-{i - 1}"
        axs[5].plot(epsilon_history[:, i], label=label)

    axs[5].legend()

    plt.tight_layout()
    plt.pause(0.05)
