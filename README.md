# Introduction

A Python library for implementing **affective active inference** models with valence and arousal dynamics. The library enables hierarchical active inference, incorporating metacognition, emotional state modeling, and a Wundt curve for valence-arousal mapping.

This work is based on the [paper](https://doi.org/10.1162/neco_a_01341).

For more information about the underlying theory, the folder Thesis contains the complete reasearch.

## Installation

To install directly from GitHub:

```bash
pip install git+https://github.com/LucaAnnese/emdeep.git
```

## Example
from emdeep.Affect_agent import affect_agent

The agent consists of two layers: the lower layer can be described as PYMDP agent and its setup is equivalent, while the higher level leverages on its inner quantities.
The agent's lower and higher levels need to be initialized as follows:
```
low_level = pymdp.Agent(A = A_l, B = B_l, inference_algo='MMP')

```
Where the inference_algo MUST BE MMP.

```
affective_agent = affect_agent(perceptive_agent=low_level, A_h = A_h, B_h = B_h, D_h = D_h, beta_rate=beta_rate, context_hidden_states=context_hidden_states, keep_history=True)
```
A straigth usage of the agent, follows the action-perception-metacognition loop:

```
start trial:
        agent.initialize()
        start timepoint:
            agent.infer()
            agent.act()
            agent.observe()
            agent.update()
        end timepoint
        agent.metacognitive_computation()
        agent.updates()
    end trial
```
# Quickstart

This example is based on the **T-Maze** environment as desribed in [here](https://github.com/infer-actively/pymdp/tree/master). 

All the matrices are set up to the needed dimensionality, but need to be initialized with the wanted distributions.

```
import numpy as np
from affective_agent.affect_agent import affect_agent
from pymdp.agent import Agent
from pymdp import utils
from pymdp.envs import TMazeEnvNullOutcome
from reward_aversion_curvature import wundt

# Step 1: Define high-level matrices for valence, arousal, and context

A_h[0] = np.zeros((len(rate_parameter), len(valence), 1)) 
A_h[1] = np.zeros((len(reward_loc_obs_high), len(reward_loc), 1))

B_h[0] = np.zeros((len(valence), len(valence), 1))
B_h[1] = np.zeros((len(reward_loc), len(reward_loc), 1))
B_h[2] = np.zeros((len(valence), len(valence), 1))

D_h[0] = np.array([0.5, 0.5])
D_h[1] = np.array([0.5, 0.5])
D_h[2] = np.array([0.5, 0.5])

# Step 2: Define a perceptive (low-level) agent
location = ['CENTER', 'LEFT ARM', 'RIGHT ARM', 'CUE LOCATION']
food = ['Left', 'Right']
num_states_low = [len(location), len(food)]

A_cue = np.zeros((len(cue_obs), len(location), len(food)))
A_reward_loc = np.zeros((len(reward_obs), len(location), len(food)))
A_l[1] = A_reward_loc
A_l[2] = A_cue

B_location = np.zeros((len(location), len(location), 4))
B_l[0] = B_location
B_trial = np.zeros((len(food), len(food), 1))
B_trial[:, :, 0] = np.eye(len(food))
B_l[1] = B_trial

low_level_agent = Agent(A=A_l, B=B_l, policy_len=2, inference_horizon=2, inference_algo='MMP')

# Step 3: Initialize the affective agent
beta_rate = np.array([[0.5, 2.0]])  # Beta for high-level valence states
context_hidden_states = [1]  # Reward location
affective_agent = affect_agent(
    perceptive_agent=low_level_agent, 
    A_h=A_h, B_h=B_h, D_h=D_h, 
    beta_rate=beta_rate, 
    context_hidden_states=context_hidden_states
)

env_low = TMazeEnvNullOutcome(reward_probs=reward_probabilities)

# Step 4: Simulate the agent
for trial in range(trials):
  affective_agent.initialize() 
  obs_env = env_low.reset(state=env_low_state)   
  
  for t in range(T):   
    action = affective_agent.step(obs_env) # AcI action-perception cycle
    obs_env= env_low.step(action)          # perform action and get new observation
    
  affective_agent.metacognition()            
  affective_agent.update()

```

# Emotional mapping

The model maps valence and arousal onto an emotional circumplex, which helps visualize the inner state of the agent in more significant way. Valence gets mapped in a linear way while arousal follows a Wundt curve, whose parameters can be redefined at needs. 

![Mapping of the valence-arousal space to emotion labels](https://github.com/user-attachments/assets/b3a346b0-8ec3-4d09-b22f-c3ff42bf91ec)

## **Features**

- **Affective Inference**: Models valence and arousal dynamics within an active inference framework.
- **Hierarchical Architecture**: Combines perceptive (lower) and metacognitive (higher) levels.
- **Customizable Models**: Easily modify priors, transitions, and parameters.
- **Visualization Tools**: Generate insights into valence, arousal, and affective charge.

## Documentation

- **affect_agent** Class: Combines a perceptive agent with a metacognitive level.
- Functions:
  - initialize: Prepares the agent for inference.
  - step: Performs the action-perception loop.
  - metacognition: Updates high-level beliefs and emotional states.
  - update: Resets and reinitializes priors for the next trial.
- Modules:
  - reward_aversion_curvature.py: Implements the Wundt curve for valence-arousal dynamics.


