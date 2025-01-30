import pymdp
import numpy as np
from pymdp import utils
from pymdp.maths import spm_log_single, softmax_obj_arr
from pymdp.control import softmax, calc_states_info_gain
from reward_aversion_curvature import wundt

def kl_div(P,Q):
    # from PYMDP 
    """
    Parameters
    ----------
    P : Categorical probability distribution
    Q : Categorical probability distribution

    Returns
    -------
    The KL-divergence of P and Q

    """
    EPS_VAL = 1e-16
    dkl = 0
    for i in range(len(P)):
        dkl += np.dot(P[i], np.log(P[i] + EPS_VAL) - np.log(Q[i] + EPS_VAL))
    return(dkl)


class affect_agent():
    def __init__(self, perceptive_agent, 
                 beta_rate, 
                 A_h,
                 B_h,
                 D_h,
                 keep_history= False,
                 context_hidden_states = None,
                  **kwargs ):
        super().__init__(**kwargs)
        #check for dimensionalities withing A, B, D and the lower level of the hierarchy

        if perceptive_agent.inference_algo == "VANILLA":
            print("Warning: the \'inference_algo\' of perceptive_agent, MUST be set to \'MMP\' to compute the Affective Charge term")

        self.low_level = perceptive_agent
        self.keep_history = keep_history
        self.beta_rate = beta_rate
        

        """
        Metacognitive level:
        Three hidden states: valence, context, arousal
        """
        high_level = { "A" : A_h,
                       "B" : B_h,
                       "D" : D_h,
                       "qs" : None}
        self.high_level = high_level
        # check for the dimensionality



        """Assure that qs_hsit is used"""
        if not hasattr(self.low_level, 'qs_hist'):
            self.low_level.save_belief_hist = True
            self.low_level.qs_hist = []
        
        # index of the context state emplyed in higher-level compuatations+ùùùùè
        if context_hidden_states is None:
            raise ValueError("context_hidden_states must be provided")
        
        #if max(context_hidden_states) > perceptive_agent.num_factors:
        #    raise ValueError("Context hidden states must be within the range of the perceptive agent's states")

        self.context_hidden_states = context_hidden_states

        self.AC = []
        self.policy_hist = []
        self.strongest_policy_prior = []
        self.gamma_hist = []

        if self.low_level.pD is None:
            pD = utils.dirichlet_like(self.low_level.D, scale=1.0)
            self.low_level.pD = pD

        self.current_policy = 0
        self.vw = []
        self.ig = []
        self.FE = []

        #self.qs_low = None

    def initialize(self):
        """
        Initialize the agent: obtain from the high level the initial beliefs over the hidden states of the lower level (priors)
        """
        #Initialize high_level qs to be a uniform distribution over all states
        if self.high_level["qs"] is None:
            num_states = [self.high_level["B"][f].shape[0] for f in range(len(self.high_level["B"]))]
            qs = utils.obj_array_uniform(num_states)
            self.high_level["qs"] = qs

        #Saving the initial belief of the agent pre-trial to be emplyed in ascending_messages (only works for MMP)
        algo = self.low_level.inference_algo
        if self.low_level.qs is None :
            self.init_belief = self.low_level.D
        if algo == "MMP":
            self.init_belief = self.low_level.qs[self.current_policy][-1] # extract last timepoint
            if self.init_belief is None:
                self.init_belief = self.low_level.qs[self.current_policy][0] #extract a random one, it's the same initially
        else:
             self.init_belief = self.low_level.qs

        # Set pre-trial prior
        empirical_prior, gamma = self.__descending_messages()

        self.low_level.gamma = gamma
        self.gamma_hist.append(self.low_level.gamma)
        #self.low_level.D = empirical_prior # che D sia empirical prior credo sia corretto, lo è 
        self.low_level.update_D(empirical_prior)
        #self.low_level.reset(init_qs = empirical_prior)
        
       

    def __ascending_messages(self):
        """Compute arousal (action) evidence"""
        posterior = self.low_level.qs[self.current_policy][-1][self.context_hidden_states] # extract post-trial beliefs
        information_gain = kl_div(posterior, self.low_level.D[self.context_hidden_states]) # arousal_potential
        #arousal = 1 / (1 + np.exp(-np.log(information_gain)))
        """compute Wundt curve elements"""
        valence, _, _ = wundt(information_gain)
        self.vw.append(valence)
        """Bayesian surprise about states as arousal indicator"""   
        ig = calc_states_info_gain(self.low_level.A, self.qs_low[self.current_policy]) # simlar to the information gain

        arousal = 1 / (1 + np.exp(-(information_gain)))
        arousal_evidence = [1 - arousal, arousal]
        self.ig.append(arousal)
        print(f"IG: {arousal}")
        """compute valence (affective) evidence"""
        beta_h = 1 / self.low_level.gamma
        """Affect hiddens state is suppose to ALWAYS be the first state in the high level"""
        prior_a = np.dot(spm_log_single(self.high_level["B"][0][:,:,0]), self.high_level["qs"][0]) 
        likelihood_a = np.dot(spm_log_single((self.beta_rate - self.AC[-1]) / self.beta_rate), beta_h / (beta_h - self.AC[-1]))
        affective_evidence = softmax_obj_arr(prior_a - likelihood_a)

     
        """ compute context evidence"""
        prior_c = np.dot(spm_log_single(self.high_level["B"][1][:,:,0]), self.high_level["qs"][1])
        likelihood_c = np.dot(spm_log_single(self.high_level["A"][1][:,:,0]).T, self.qs_low[self.current_policy][0][1]) # uncomment to use last belief
        #likelihood_c = np.dot(spm_log_single(self.high_level["A"][1][:,:,0]).T, self.init_belief[1]) # using first belief
        context_evidence = softmax(prior_c + likelihood_c)

        return affective_evidence, context_evidence, arousal_evidence

    def __descending_messages(self):
        """
        compute beta
        compute gamma for lower level as 1/beta
        compute empirical prior per each num_states_low"""

        # Compute policy prior (arousal related)

        #E = utils.obj_array_zeros(len(self.low_level.policies))


        # Extract high beliefs
        if self.high_level["qs"] is None:
            beliefs_high = self.high_level["D"]
        else:
            beliefs_high = self.high_level["qs"]

        # Compute the new b parameter
        b = float(np.dot(np.dot(self.beta_rate[0], (self.high_level["A"][0][:,:,0])), beliefs_high[0]))  # A_h[0]: likelihood for affective states of higher level
        a = float(np.dot(np.dot(self.beta_rate[0], (self.high_level["A"][0][:,:,0])), beliefs_high[2])) 
        #a = float(np.dot(np.dot(self.beta_rate[0], (self.high_level["A"][0][:,:,0])), [1 - self.ig[-1], self.ig[-1]])) 
        gamma = 1/((b+a)/2)
        #gamma = 1/b 
        # Compute empirical prior for the hidden states
        empirical_prior = utils.obj_array_zeros(self.low_level.num_states)
        for f in range(self.low_level.num_factors): # get only the context state, not the others
            if f in self.context_hidden_states:
                empirical_prior[f] = np.dot(beliefs_high[f], self.high_level["A"][1]).T  # if its context
                empirical_prior[f] = empirical_prior[f][0]

            else:
                empirical_prior[f] = self.low_level.D[f] # otherwise maintain the starting prior
        
        return empirical_prior, gamma


    def compute_affective_charge(self):
        
        AC = np.dot((self.prior_pi_low - self.qs_pi), self.G)
        #AC *= 10
        print(AC)
        self.AC.append(AC)
        return AC


    def step(self, obs):
        """compute the action-perception cycle"""
        qs_low = self.low_level.infer_states(obs) 
        self.qs_low = qs_low

        q_pi_low, G = self.low_level.infer_policies() #save locally these quantities
        self.qs_pi = q_pi_low
        self.prior_pi_low = softmax(G * self.low_level.gamma + spm_log_single(self.low_level.E))
        self.G = G   
        """Compute the action"""        
        action = self.low_level.sample_action()
        """Compute AC"""
        _ = self.compute_affective_charge()

        return action

    def metacognition(self):
        "Testing E update"

        # Retrieve the current policy (ONLY IF MMP)
        tmp  = self.low_level.prev_actions[-self.low_level.policy_len:]
        tmp = np.vstack(tmp).astype(int)
        index = next((i for i, array in enumerate(self.low_level.policies) if np.array_equal(array, tmp)), -1)

        # Update E: all methods are fine
        self.low_level.E = softmax(self.low_level.E * self.qs_pi)
        #self.low_level.E = softmax(self.low_level.E - self.low_level.gamma * spm_log_single(self.prior_pi_low))
        #self.low_level.E = self.prior_pi_low
        #if index != -1:
        #    self.low_level.E[index] += 1   # increment the probability of the current policy -> the value is arbitrary 
        #self.low_level.E = softmax(self.low_level.E)
       

        self.current_policy = index
        self.policy_hist.append(self.current_policy)
        " compute the metacognitive level"
        ae, ce, are = self.__ascending_messages()
        self.high_level["qs"] = np.array([ae[0], ce, are])



    def update(self):
        """low_level reset after a trial"""  
        current_qs = self.qs_low[self.current_policy]
        if (len(current_qs) > self.low_level.inference_horizon):
            self.qs_low = [item[:self.low_level.inference_horizon] for item in self.qs_low]

        self.low_level.reset(self.qs_low)
        print(f"qs_low: {self.qs_low}")
        # Reset low_level inner variables
        self.low_level.prev_actions = None
        self.low_level.prev_obs = []
        self.low_level.update_D(self.qs_low)
        beta = 1/ self.low_level.gamma
        self.low_level.gamma = 1/ (beta - self.AC[-1])

        "Update the high level"
        # context and affective evidence to be update via the B matrix
        for f in range(len(self.high_level["qs"])):
            self.high_level["qs"][f] = np.dot(self.high_level["qs"][f], self.high_level["B"][f].T)



    


    
