import factory_machines_env
from factory_machines_env.wrappers import FactoryMachinesFlattenWrapper
import gym
from gym.utils.play import play
from q_learning_agent import QLearningAgent, train_q_learning_agent, play_agent
from dqn_agent import DQNAgent, train_dqn_agent
import json
import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = FactoryMachinesFlattenWrapper(
    #     gym.make(
    #         "factory_machines/FactoryMachines-v0",
    #         render_mode="human",
    #         recipe_complexity=2
    #     )
    # )

    def env_factory(seed):
        return gym.make("CartPole-v1").unwrapped

    env = env_factory(0)

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        epsilon=1,
        device=device
    )

    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

    try:
        train_dqn_agent(
            env_factory,
            agent,
            opt
        )
    except KeyboardInterrupt:
        print("Interrupted, saving agent...")

    agent.save("./dqn_agent_params.pt")


if __name__ == '__main__':
    main()
