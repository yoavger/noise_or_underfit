import numpy as np
import pandas as pd
from utils import *

def hybrid_sim(param, num_of_trials, transition_prob, reward_probs):
    """
    this funcation simulate an hybrid  agent in the *reduce* two step task 
    Args:
        param: parameters of the agent 
        num_of_trials: number of trials of the simulation
        transition_prob: a matrix 2*2 of the transition function front the first stage to the second stage 
        reward_probs: a matrix num_of_trials*2*2 of the probability for reward of both second stage states for all trials
    
    Returns:
        df: DataFrame of the behavior of the agent in the simulation
        
    action are coded: 
        0 and 1 
    state are coded: 
        0 - first stage
        1 - second stage first state
        2 - second stage second state
    """

    # set up parameters 
    alpha = param['alpha']
    beta = param['beta']
    w = param['w']
    eligibility_trace = 0.9 # keep constant to all agent  

    # q_values of model free
    # index 0 state 
    # index 1 action
    q_mf = np.zeros(shape=(3,2))
    # q_values of model based
    q_mb = np.zeros(2)
    # the weighted sum of model-based and model-free values
    q_net = np.zeros(2)

    state_transition_mat = np.array(
        [[2,1],
        [1,2]]
    )

    # store data from each trial 
    data = DataOfSim(num_of_trials)

    for t in range(num_of_trials):

        # Q_model-based values of the first stage actions - Bellman’s equation 
        q_mb[0] = (transition_prob[0,0]*q_mf[1,0]) + (transition_prob[0,1]*q_mf[2,0])
        q_mb[1] = (transition_prob[1,0]*q_mf[1,0]) + (transition_prob[1,1]*q_mf[2,0])

        # weighted sum of model-based and model-free action values
        q_net[0] = (w*q_mb[0]) + ((1-w)*q_mf[0,0])  
        q_net[1] = (w*q_mb[1]) + ((1-w)*q_mf[0,1])

        # calculate probability for choosing action 0 in the for first stage with softmax choice rule 
        prob_0 = np.exp( beta*(q_net[0])) / np.sum( np.exp(beta*q_net) ) 

        # choose action_1 according to the probability of the first stage
        action_1 = np.random.choice([0,1] , p=[prob_0, 1-prob_0])

        # sample a transation type
        transition_type = np.random.choice([0,1], p = transition_prob[1,:]) # 0 = rare / 1 = common
        # transation to second stage according to action and transation type
        state = state_transition_mat[action_1,transition_type]

        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=reward_probs[t,state-1,:]) 

        # calculate prediction error
        p_e_1 = q_mf[state,0] - q_mf[0,action_1] 
        p_e_2 = reward - q_mf[state,0] 

        # update q_mf according to q_learning formula
        q_mf[0,action_1] = q_mf[0,action_1] + alpha*p_e_1 + eligibility_trace*(alpha*p_e_2) 
        q_mf[state,0] = q_mf[state,0] + alpha*p_e_2

        # stroe data of the trial
        data.action_1_list[t] = action_1
        data.stage_2_state[t] = state - 1
        data.transition_list[t] = transition_type
        data.reward_list[t] = reward
        data.probs[t] = prob_0
        data.delta_q[t] = q_net[0] - q_net[1]
       
    df = pd.DataFrame(data.createDic())
    return df
