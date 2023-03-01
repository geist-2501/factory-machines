from typing import Dict
from operator import itemgetter

from factory_machines.dqn_agent import DQN
from talos import Agent


class HDQNAgent(Agent):

    def __init__(
            self,
            obs_size: int,
            n_goals: int,
            n_actions: int,
            device: str = 'cpu'
    ) -> None:
        super().__init__("h-DQN")

        self.meta_cont_net = DQN(obs_size, n_goals).to(device)  # Meta-controller net / Q2.
        self.meta_cont_net_fixed = DQN(obs_size, n_goals).to(device)  # Meta-controller fixed net.
        self.cont_net = DQN(n_goals, n_actions).to(device)  # Controller net / Q1.
        self.cont_net_fixed = DQN(n_goals, n_actions).to(device)  # Controller fixed net.

    def forward(self, state):
        # TODO this is non-trivial.
        pass

    def get_action(self, state):
        pass

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
