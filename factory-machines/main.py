import factory_machines_env
from factory_machines_env.wrappers import FactoryMachinesFlattenWrapper
import gym
from q_learning_agent import QLearningAgent, train_q_learning_agent, play_agent
import json


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = FactoryMachinesFlattenWrapper(
        gym.make(
            "factory_machines/FactoryMachines-v0",
            render_mode="human",
            recipe_complexity=2
        )
    )

    agent = QLearningAgent(
        alpha=0.5,
        epsilon=1,
        discount=0.99,
        action_space=env.action_space
    )

    try:
        train_q_learning_agent(env, agent, n_episodes=800)
    except KeyboardInterrupt:
        print("Interrupted, saving agent...")

    # TODO save agent.

    play_agent(env, agent)


if __name__ == '__main__':
    main()
