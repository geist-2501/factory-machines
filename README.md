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

## Installing and Running

To run the project, first install the dependencies. Start by creating a venv.

```
python -m venv .venv
```

Then activate it using `source ./.venv/bin/activate` on Linux or `.\.venv\Scripts\activate`.
Installing dependencies is a little fiddly. Since the project uses PyTorch GPU acceleration, you must either; 

 - Install Cuda 11.7 yourself and execute `pip install -r requirements.txt`.
 - Change the dependencies to install CPU-based Torch binaries (torch==1.13.1+cu117 => torch==1.13.1+cpu), then install via the above.

Then enjoy using Talos!

```
python main.py -v
python main.py profile train talos_profiles.yaml debug_hdqn_map_0
python main.py talfile replay agents_form_eval/final_dqn_map_2.tal
```


