# Factory Machines

The code repo for my CS5199 Master’s dissertation.

The `./factory_machines/agents` directory contains;
 
 - A DQN implementation.
 - A h-DQN implementation.
 - A Q-Learning implementation.
 - A Nearest-Neighbour heuristic.
 - A really (intentionally) bad heuristic.
 - A simultaneous localisation and mapping A* system.

The `./factory_machines/envs` directory contains;

 - The Factory Machines (fm_env_multi.py) environment used as a testbed for my thesis.
 - A simplified version for debugging (fm_env_single.py).
 - A discrete-stochastic env recreation, used in the h-DQN paper by Kulkarni et al. (2016).
 - Some other debugging envs.

The `./talos` directory contains the Talos CLI.

The Talos CLI, DQN implementation and h-DQN implementation will be split off into their own repos once finished.
