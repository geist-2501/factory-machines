from talos import register_agent
from agents.dqn_agent import DQNAgent, dqn_training_wrapper
from agents.fm_hdqn_agent import FactoryMachinesHDQNAgent
from agents.ds_mpd_hdqn_agent import DiscreteStochasticHDQNAgent
from agents.h_dqn_agent import hdqn_training_wrapper

register_agent(
    id="DQN",
    agent_factory=lambda obs, n_actions, device: DQNAgent(obs, n_actions, device=device),
    training_wrapper=dqn_training_wrapper
)

register_agent(
    id="FM-HDQN",
    agent_factory=lambda obs, n_actions, device: FactoryMachinesHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper
)

register_agent(
    id="DS-HDQN",
    agent_factory=lambda obs, n_actions, device: DiscreteStochasticHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper
)