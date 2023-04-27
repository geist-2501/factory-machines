from factory_machines.agents.dqn_agent import DQNAgent, dqn_training_wrapper, dqn_graphing_wrapper
from factory_machines.agents.ds_mpd_hdqn_agent import DiscreteStochasticHDQNAgent
from factory_machines.agents.fm_hdqn_agent import FactoryMachinesHDQNAgent
from factory_machines.agents.fm_masked_hdqn_agent import FactoryMachinesMaskedHDQNAgent
from factory_machines.agents.gw_hdqn_agent import GridWorldHDQNAgent
from factory_machines.agents.h_dqn_agent import hdqn_training_wrapper, hdqn_graphing_wrapper
from factory_machines.agents.heuristics.aisled_nn_agent import AisledNNAgent
from factory_machines.agents.heuristics.fm_simple_heuristic_agent import FMSimpleHeuristicAgent
from factory_machines.agents.heuristics.nn_agent import NNAgent
from talos import register_agent

register_agent(
    agent_id="DQN",
    agent_factory=lambda obs, n_actions, device: DQNAgent(obs, n_actions, device=device),
    graphing_wrapper=dqn_graphing_wrapper,
    training_wrapper=dqn_training_wrapper
)

register_agent(
    agent_id="FM-HDQN",
    agent_factory=lambda obs, n_actions, device: FactoryMachinesHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper,
    graphing_wrapper=hdqn_graphing_wrapper
)

register_agent(
    agent_id="FM-HDQN-masked",
    agent_factory=lambda obs, n_actions, device: FactoryMachinesMaskedHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper,
    graphing_wrapper=hdqn_graphing_wrapper
)

register_agent(
    agent_id="GW-HDQN",
    agent_factory=lambda obs, n_actions, device: GridWorldHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper,
    graphing_wrapper=hdqn_graphing_wrapper
)

register_agent(
    agent_id="DS-HDQN",
    agent_factory=lambda obs, n_actions, device: DiscreteStochasticHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper
)

register_agent(
    agent_id="FM-Highest",
    agent_factory=lambda obs, n_actions, device: FMSimpleHeuristicAgent(obs, n_actions, device),
)

register_agent(
    agent_id="FM-NN",
    agent_factory=lambda obs, n_actions, device: NNAgent(obs, n_actions, device),
)

register_agent(
    agent_id="FM-AisledNN",
    agent_factory=lambda obs, n_actions, device: AisledNNAgent(obs, n_actions, device),
)