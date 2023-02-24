from talos import register_agent
from factory_machines.dqn_agent import DQNAgent, dqn_training_wrapper

register_agent(
    id="DQN",
    agent_factory=lambda n_states, n_actions, device: DQNAgent(n_states, n_actions, device=device),
    training_wrapper=dqn_training_wrapper
)
