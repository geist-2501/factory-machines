from talos import register
from factory_machines.dqn_agent import DQNAgent, dqn_training_wrapper

register(
    id="DQN",
    agent_factory=lambda n_states, n_actions: DQNAgent(n_states, n_actions),
    training_wrapper=dqn_training_wrapper
)
