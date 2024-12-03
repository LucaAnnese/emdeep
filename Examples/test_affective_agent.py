import copy
import numpy as np
import matplotlib.pyplot as plt

from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import softmax_obj_arr, spm_log_single
from pymdp.envs import TMazeEnvNullOutcome

from pymdp import agent
from Affect_agent import affect_agent
from MetacognitiveLevel import MetacognitiveLevel



def matrices_hl():
    """
    The A, B, D matrices for the higher level are built.
    """

    """ Likelihood mapping """
    # This is the affective prior, a.k.a. likelihood uncertainty
    uncertainty = 0.03
    A_affective = np.zeros((len(rate_parameter), len(valence), 1))  # non devo aggiungere context state ??
    A_affective[:, :, 0] = np.array([[1.0 - uncertainty, uncertainty], [uncertainty, 1.0 - uncertainty]])
    A_h[0] = A_affective

    # This is the contextual prior, a.k.a. likelihood from context states to lower level
    A_reward_loc = np.zeros((len(reward_loc_obs_high), len(reward_loc), 1))
    A_reward_loc[:, :, 0] = np.eye(len(reward_loc))
    A_h[1] = A_reward_loc

    """ State transitions """
    B_valence = np.zeros((len(valence), len(valence), 1))
    B_valence[:, :, 0] = np.array([[.8, .3], [.2, .7]])
    B_h[0] = B_valence

    B_context = np.zeros((len(reward_loc), len(reward_loc), 1))
    B_context[:, :, 0] = np.array([[.8, .2], [.2, .8]])
    B_h[1] = B_context

    # keep valence values
    B_arousal = np.zeros((len(valence), len(valence), 1))
    B_arousal[:, :, 0] = np.array([[1, 0], [0, 1]])
    B_h[2] = B_arousal

    """ Initial state prior """
    D_valence = np.array([0.5, 0.5])
    D_h[0] = D_valence

    D_context = np.array([0.5, 0.5])
    D_h[1] = D_context

    D_arousal = np.array([0.5, 0.5])
    D_h[2] = D_arousal


trials = 64
T = 2 # number of timepoints per trial

""" Higher-level MDP for affective inference """
# hidden states
valence = ['Positive', 'Negative']  # affective state
reward_loc = ['Left', 'Right']  # context state

rate_parameter = ['0.5', '2.0']  # beta - positive valence: 2.0 - negative valence: 0.5
reward_loc_obs_high = ['Left', 'Right']
beta_rate = np.zeros((1, 2))
beta_rate[0] = np.array([0.5, 2.0])

num_states_high = [len(valence), len(reward_loc)]  # number of hidden states
num_factors_high = len(num_states_high)  # number of hidden state factors
num_obs_high = [len(rate_parameter), len(reward_loc_obs_high)]  # number of outcomes
num_modalities_high = len(num_obs_high)  # number of outcome factors

# matrix high level
A_h = utils.obj_array(num_modalities_high)
B_h = utils.obj_array(num_factors_high+1)
D_h = utils.obj_array(num_factors_high+1)

""" Lower-level MDP for context-specific active inference """
location = ['CENTER', 'LEFT ARM', 'RIGHT ARM', 'CUE LOCATION']
food = ['Left', 'Right']
reward_obs = ['No reward', 'Reward!', 'Loss!']
cue_obs = ['No Cue', 'Cue Left', 'Cue Right']

num_states_low = [len(location), len(food)]
num_factors_low = len(num_states_low)
num_obs_low = [len(reward_obs), len(cue_obs)]

# policy vector - allowable combinations of actions
policy_len = 2


"""
    Affective inference loop

    The agent is initialized and the affective inference loop is run for each trial.
    
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

    """
reward_probabilities = [1.0, 0.0]
env_low = TMazeEnvNullOutcome(reward_probs=reward_probabilities)
A_gp = env_low.get_likelihood_dist()
B_gp = env_low.get_transition_dist()
A_l, B_l = copy.deepcopy(A_gp), copy.deepcopy(B_gp)


"""Define perceptive agent"""
#A_l, B_l, C_l, D_l, E_l, pD_l = matrices_ll(env_low)
A_cue = np.zeros((len(cue_obs), len(location), len(food)))
A_cue[:, :, 0] = np.array([[1, 1, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]])
A_cue[:, :, 1] = np.array([[1, 1, 1, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]])

A_reward_loc = np.zeros((len(reward_obs), len(location), len(food)))
A_reward_loc[:, :, 1] = np.array([[1, 0, 0, 1],
                                  [0, 0.02, 0.98, 0],
                                  [0, 0.98, 0.02, 0]])
A_reward_loc[:, :, 0] = np.array([[1, 0, 0, 1],
                                  [0, 0.98, 0.02, 0],
                                  [0, 0.02, 0.98, 0]])
A_l[1] = A_reward_loc
A_l[2] = A_cue
"""Trying to make it hard coded"""
B_location = np.zeros((len(location), len(location), 4))
B_location[:, :, 0] = np.array([[1, 0, 0, 1],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0]])
B_location[:, :, 1] = np.array([[0, 0, 0, 0],
                                [1, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0]])
B_location[:, :, 2] = np.array([[0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [1, 0, 1, 1],
                                [0, 0, 0, 0]])
B_location[:, :, 3] = np.array([[0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 1]])
B_l[0] = B_location
B_trial = np.zeros((len(food), len(food), 1))
B_trial[:, :, 0] = np.eye(len(food))
B_l[1] = B_trial
low_level = Agent(A = A_l, B = B_l, policy_len=2, inference_algo='MMP',policy_sep_prior=False, use_BMA=True,
                  inference_horizon=2, save_belief_hist=True, use_param_info_gain=True,
                 sampling_mode='full', action_selection="stochastic")
print(low_level.policies)

#low_level.E.fill(2.3)
low_level.D[0] = utils.onehot(0, low_level.num_states[0])
low_level.D[1] = np.array([0.5, 0.5])
low_level.C[1][1] = 4.0
low_level.C[1][2] = -6.0 

"""Define metacognitive level"""
matrices_hl()


"""Define affective agent"""
context_hidden_states = [1]
affective_agent = affect_agent(perceptive_agent=low_level, A_h = A_h, B_h = B_h, D_h = D_h, beta_rate=beta_rate, context_hidden_states=context_hidden_states, keep_history=True)
food = ['Left', 'Right']
obs_env = env_low.reset()    # reset environment
env_low_state = env_low._state
env_switch = copy.deepcopy(env_low_state)
env_switch[1] = env_low._state[1][::-1]

history_D = []
history_policy = []
history_ae = []


for trial in range(trials):
    print(f"Trial {trial}")
    affective_agent.initialize() # get priors
    
    h_action = []
    if trial < trials/2:
        obs_env = env_low.reset(state=env_low_state)    # reset environment
    else:
        obs_env = env_low.reset(state=env_switch)    # reset environment
    #print(f"Starting beleifs: {affective_agent.low_level.qs}")
    for t in range(T):   
        
        
        action = affective_agent.step(obs_env) # AcI action-perception cycle
        h_action.append(int(action[0])) 
        print(f"action: {location[int(action[0])]}")
        obs_env= env_low.step(action)          # perform action and get new observation
        print(f"Observation: {location[obs_env[0]], reward_obs[obs_env[1]], cue_obs[obs_env[2]]}")

    
    

    #print(f"Action: {h_action[0], h_action[1]}")
    history_D.append(affective_agent.low_level.D[1][0] )
    affective_agent.metacognition()            
    history_ae.append(affective_agent.high_level["qs"][0][0])
    history_policy.append(affective_agent.current_policy)
    affective_agent.update()



plt.figure(figsize=(15, 6))
plt.plot(np.arange(2*trials), affective_agent.AC, '-o')
plt.axvline(x=trials, color='r', linestyle='--')
plt.title("Affective charge (with-in trials)")
plt.figure()
plt.plot(np.arange(trials), affective_agent.gamma_hist)
plt.axvline(x=trials/2, color='r', linestyle='--')
plt.title("Gamma - Model confidence")
plt.figure()
plt.plot(np.arange(trials), history_D)
plt.axvline(x=trials/2, color='r', linestyle='--')
plt.title("Beliefs on reward location (pre-trial)")
plt.yticks([0.2, 0.5, 0.8], ['Left', "50/50", 'Right'])
plt.figure(figsize=(15, 6))
plt.plot(np.arange(trials), history_ae, '-o')
plt.plot(np.arange(trials), affective_agent.ig)
plt.axvline(x=trials/2, color='r', linestyle='--')
plt.plot(np.arange(trials), affective_agent.vw)

#plt.plot(np.arange(trials), affective_agent.F)
plt.legend(["Valence", "Arousal", "Context change", " WundtMapping"])

plt.figure()
#plt.plot(np.arange(trials), affective_agent.FE)



fig, axes = plt.subplots(3, 1, figsize=(15, 12))  # 3 rows, 1 column

# Plot 1: Affective charge (within trials)
axes[0].plot(np.arange(2*trials), affective_agent.AC, '-o')
axes[0].axvline(x=trials, color='r', linestyle='--')
axes[0].set_title("Affective charge (within trials)", fontsize=16)
axes[0].set_ylabel("AC", fontsize=14)  # Y-axis label

# Plot 2: Gamma - Model confidence
axes[1].plot(np.arange(trials), affective_agent.gamma_hist)
axes[1].axvline(x=trials/2, color='r', linestyle='--')
axes[1].set_title("Gamma - Model confidence", fontsize=16)
axes[1].set_ylabel(r"$\gamma$", fontsize=14)  # Y-axis label

# Plot 3: Beliefs on reward location (pre-trial)
axes[2].plot(np.arange(trials), history_D)
axes[2].axvline(x=trials/2, color='r', linestyle='--')
axes[2].set_title("Beliefs on reward location (pre-trial)", fontsize=16)
axes[2].set_ylabel(r"$D_{\text{reward}}$", fontsize=14)  # Y-axis label
axes[2].set_yticks([0.2, 0.5, 0.8])
axes[2].set_yticklabels(['Left', "50/50", 'Right'])

# Adjust layout and add common x-axis label
fig.text(0.5, 0.04, 'Trials', ha='center', fontsize=12)  # X-axis label at the bottom

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to include the text
plt.show()




"""Emotional cimcumplex mapping"""
arousal = affective_agent.vw
valence = [2*x - 1 for x in history_ae]

colors_dict = {'Happy': 'palegreen', 
          'Excited': 'lightyellow', 
          'Angry': 'orange', 
          'Frustated': 'orangered', 
          'Sad': 'pink', 
          'Bored': 'purple',
          'Sleepy': 'blue', 
          'Relaxed': 'cyan', 
          'A': 'white'}

def emotional_circumplex(X, Y):

    assert len(X) == len(Y), "Valence and Arousal must have the same length"
    emotions = []
    for x, y in zip(X, Y):
        if x > 0 and y > 0:
            if y <= x:
                emotions.append('Happy')
            else:
                emotions.append('Excited')
        elif x < 0 and y > 0:
            if y > -x:
                emotions.append('Angry')
            else:
                emotions.append('Frustated')
        elif x < 0 and y < 0:
            if y >= x:
                emotions.append('Sad')
            else:
                emotions.append('Bored')
        elif x > 0 and y < 0:
            if y < -x:
                emotions.append('Sleepy')
            else:
                emotions.append('Relaxed')
        else:
            # Punto sull'asse o all'origine (considerato spicchio A per convenzione)
            emotions.append('A')

    return emotions

emotions = emotional_circumplex(arousal,valence )
colors = [colors_dict[emotion] for emotion in emotions]

plt.figure(figsize=(15, 6))
for i, c in enumerate(colors):
    plt.axvspan(i, i+1, color=c)

plt.title("Agent's emotional state", fontsize=16)
plt.xlabel("Trials", fontsize=12)


used_colors = set(colors)  # colors used in the plot

import matplotlib.patches as mpatches

# Create legend entries only for the colors used
legend_patches = [
    mpatches.Rectangle((0, 0), 1, 1, color=color, edgecolor='black', linewidth=2)  # Rectangle for each entry
    for emotion, color in colors_dict.items() if color in used_colors
]

# Add the legend to the plot with a box and bold edges for each entry
plt.legend(
    handles=legend_patches,
    labels=[emotion for emotion, color in colors_dict.items() if color in used_colors],
    title="Emotional States",
    loc="upper right",
    frameon=True,  # Ensure the frame is on
    edgecolor='black',  # Set the edge color of the legend box
    facecolor='white',  # Set the background color of the legend box
    title_fontsize='14',  # Optional: title font size
    fontsize='12'  # Optional: label font size
)

plt.xticks(range(len(colors)),[f"{i}" for i in range(len(colors))])
plt.xticks(range(0,64,10))
plt.show()



fig, axes = plt.subplots(4, 1, figsize=(15, 14))  # 3 rows, 1 column

# Plot 1: Valence\Arousal
axes[0].plot(np.arange(trials), history_ae, label='Valence')
axes[0].plot(np.arange(trials), affective_agent.ig, label='Arousal')
axes[0].axvline(x=trials/2, color='r', linestyle='--')
axes[0].set_title("Valence\\Arousal Trend", fontsize=16)
axes[0].legend(["Valence", "Arousal"], loc='upper right', fontsize=12, borderpad=1, handlelength=2.5, frameon=True, fancybox=True)
# Plot 2: Arousal Mapped
axes[1].plot(np.arange(trials), affective_agent.ig, label='Aroused Probability')
axes[1].plot(np.arange(trials), arousal, label='Arousal Value')
axes[1].axvline(x=trials/2, color='r', linestyle='--')
axes[1].set_title("Aroused level mapped into value", fontsize=16)

# Plot 3: Valece Mapped
axes[2].plot(np.arange(trials), history_ae, label='Valenced Probability')
axes[2].plot(np.arange(trials), valence, label='Valence Value')
axes[2].axvline(x=trials/2, color='r', linestyle='--')
axes[2].set_title("Valenced level mapped into value", fontsize=16)

# Plot 4: Emotional State
for i, c in enumerate(colors):
    axes[3].axvspan(i, i+1, color=c)

axes[3].set_title("Agent's emotional state", fontsize=16)

used_colors = set(colors)  # colors used in the plot
import matplotlib.patches as mpatches

# Create legend entries only for the colors used
legend_patches = [
    mpatches.Rectangle((0, 0), 1, 1, color=color, edgecolor='black', linewidth=2)  # Rectangle for each entry
    for emotion, color in colors_dict.items() if color in used_colors
]

# Add the legend to the plot with a box and bold edges for each entry
axes[3].legend(
    handles=legend_patches,
    labels=[emotion for emotion, color in colors_dict.items() if color in used_colors],
    title="Emotional States",
    loc="upper right",
    frameon=True,  # Ensure the frame is on
    edgecolor='black',  # Set the edge color of the legend box
    facecolor='white',  # Set the background color of the legend box
    title_fontsize='14',  # Optional: title font size
    fontsize='12'  # Optional: label font size
)

axes[3].set_xticks(range(len(colors)),[f"{i}" for i in range(len(colors))])
axes[3].set_xticks(range(0,64,10))
# Adjust layout and add common x-axis label
fig.text(0.5, 0.04, 'Trials', ha='center', fontsize=12)  # X-axis label at the bottom

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to include the text
plt.show()

