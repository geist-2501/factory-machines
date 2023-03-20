from agents.fm_simple_heuristic_agent import FMSimpleHeuristicAgent
from talos import register_agent
from agents.dqn_agent import DQNAgent, dqn_training_wrapper, dqn_graphing_wrapper
from agents.fm_hdqn_agent import FactoryMachinesHDQNAgent
from agents.ds_mpd_hdqn_agent import DiscreteStochasticHDQNAgent
from agents.h_dqn_agent import hdqn_training_wrapper


register_agent(
    agent_id="DQN",
    agent_factory=lambda obs, n_actions, device: DQNAgent(obs, n_actions, device=device),
    graphing_wrapper=dqn_graphing_wrapper,
    training_wrapper=dqn_training_wrapper
)

register_agent(
    agent_id="FM-HDQN",
    agent_factory=lambda obs, n_actions, device: FactoryMachinesHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper
)

register_agent(
    agent_id="DS-HDQN",
    agent_factory=lambda obs, n_actions, device: DiscreteStochasticHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper
)

register_agent(
    agent_id="FM-Simple",
    agent_factory=lambda obs, n_actions, device: FMSimpleHeuristicAgent(obs, n_actions, device),
)
