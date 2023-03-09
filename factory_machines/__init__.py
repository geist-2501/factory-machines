from talos import register_agent
from factory_machines.dqn_agent import DQNAgent, dqn_training_wrapper
from factory_machines.fm_hdqn_agent import FactoryMachinesHDQNAgent
from factory_machines.h_dqn_agent import hdqn_training_wrapper

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