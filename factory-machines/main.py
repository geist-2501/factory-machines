import factory_machines_env
import gym
from gym.utils import play
import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make("factory_machines/FactoryMachines-v0", render_mode="rgb_array")
    s, _ = env.reset()
    print(s)
    play.play(env)


if __name__ == '__main__':
    main()
