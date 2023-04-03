from agents.dqn_agent import DQNAgent, dqn_training_wrapper, dqn_graphing_wrapper, dqn_agent_wrapper
from agents.ds_mpd_hdqn_agent import DiscreteStochasticHDQNAgent
from agents.fm_hdqn_agent import FactoryMachinesHDQNAgent, fm_hdqn_agent_wrapper
from agents.gw_hdqn_agent import GridWorldHDQNAgent
from agents.h_dqn_agent import hdqn_training_wrapper, hdqn_graphing_wrapper
from agents.heuristics.aisled_nn_agent import AisledNNAgent
from agents.heuristics.fm_simple_heuristic_agent import FMSimpleHeuristicAgent
from agents.heuristics.nn_agent import NNAgent
from talos import register_agent

register_agent(
    agent_id="DQN",
    agent_factory=lambda obs, n_actions, config, device: dqn_agent_wrapper(obs, n_actions, config, device=device),
    graphing_wrapper=dqn_graphing_wrapper,
    training_wrapper=dqn_training_wrapper
)


register_agent(
    agent_id="FM-HDQN",
    agent_factory=lambda obs, n_actions, config, device: fm_hdqn_agent_wrapper(obs, n_actions, config, device),
    training_wrapper=hdqn_training_wrapper,
    graphing_wrapper=hdqn_graphing_wrapper
)

register_agent(
    agent_id="GW-HDQN",
    agent_factory=lambda obs, n_actions, config, device: GridWorldHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper,
    graphing_wrapper=hdqn_graphing_wrapper
)

register_agent(
    agent_id="DS-HDQN",
    agent_factory=lambda obs, n_actions, config, device: DiscreteStochasticHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper
)

register_agent(
    agent_id="FM-Highest",
    agent_factory=lambda obs, n_actions, config, device: FMSimpleHeuristicAgent(obs, n_actions, device),
)

register_agent(
    agent_id="FM-NN",
    agent_factory=lambda obs, n_actions, config, device: NNAgent(obs, n_actions, device),
)

register_agent(
    agent_id="FM-AisledNN",
    agent_factory=lambda obs, n_actions, config, device: AisledNNAgent(obs, n_actions, device),
)