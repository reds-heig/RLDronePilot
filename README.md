# RLDronePilot - Fully Autonomous Line-Follower Drone
This project was realized by the [REDS institute](https://reds.heig-vd.ch/) @ [HEIG-VD](https://heig-vd.ch/).

Authors: Guillaume Chacun, Mehdi Akeddar, Thomas Rieder, Bruno Da Rocha Carvalho and Marina Zapater<br>
REDS. School of Engineering and Management Vaud, HES-SO University of Applied sciences and Arts Western Switzerland<br>
Email: {guillaume.chacun, mehdi.akeddar, thomas.rieder, bruno.darochacarvalho, marina.zapater}@heig-vd.ch<br>

## Goal
This Jupyter notebook is used to train a DDPG (reinforcement learning) model to control a drone, specifically to guide it to follow a predefined line on the ground. A separate deep learning model is used to identify and track the line from images captured by the on-board camera. The pilot module processes real-world normalized coordinates of the line and outputs forward, lateral, and angular velocities for the drone.

## Content of this repository
- *train_drone_pilot.ipynb* : Notebook used to train the reinforcement learning model.
- *Agents.py* : DDPG related classes (OUActionNoise, abstract Agent class, LearningAgent, ExplorationAgent).
- *DDPGNetworks.py* : DDPG related classes (CriticNetwork, ActorNetwork).
- *SharedReplayBuffer.py* : DDPG replay buffer shared across processes.
- *Drone.py* : Class to simulate the behavior of the drone.
- *Line.py* : Class to generate a random line for the drone to follow.
- *Environment.py* : Class to handle the simulation (episodes). Contains an instance of Drone and Line.

## Setup
Install dependencies:

```bash
poetry install
```

Activate the virtual environment of Poetry:
```bash
poetry shell
```

Start your Jupyter Lab server:
```bash
jupyter lab
```

## Disclaimer
Most of the DDPG code is from [@philtabor's GitHub](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander/ddpg_torch.py) (last visited on March 5th, 2024).
