import numpy as np
import pandas as pd
from utils import *

def mb_predict(df,parameters):
    """
    this funcation predict the action of a recoverd model based agent in the *reduce* two step task 
    Args:
        paramters: parameters of the agent 
        df: DataFrame of the true behavior of the agent in the simulation we want to predict 
        
    
    Returns:
    accuracy - number of action predicted correctly (argmax) 
    choices_probs_0 - a vector of length num_of_trials of the probability of choosing action 0 in the first stage
        
    action are coded: 
        0 and 1 
    state are coded: 
        0 - first stage
        1 - second stage first state
        2 - second stage second state
    """
    # counter of the number of action classified correctly (accuracy)
    accuracy = 0 
    num_of_trials = len(df)
    choices_probs_0 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    state_of_stage_2 = list(map(int, df['state_of_stage_2']))
    reward_list = list(map(int, df['reward'])) 
 
    # set up paramters of the agent     
    alpha = parameters[0]
    beta = parameters[1]

    # initialize Q-values 
    q_mf = np.zeros(2)
    q_mb = np.zeros(2)
    
    # state transition funcation 
    transition_prob = np.array([[.8,.2],
                                [.2,.8]]) 

    for t in range(num_of_trials):

        q_mb[0] = (transition_prob[0,0]*q_mf[0]) + (transition_prob[0,1]*q_mf[1])
        q_mb[1] = (transition_prob[1,0]*q_mf[0]) + (transition_prob[1,1]*q_mf[1])

        p = np.exp( beta*(q_mb) ) / np.sum( np.exp( beta*(q_mb))) 
        
        # predict action according max probs 
        action_predict = np.argmax(p)
        choices_probs_0[t] = p[0]

        # get true state, and reward 
        state = state_of_stage_2[t] 
        reward = reward_list[t]
    
        p_e = reward - q_mf[state] 

        q_mf[state] = q_mf[state] + alpha*p_e

        # cheek if prediction match the true action
        if action_predict == action_stage_1[t]:
            accuracy+=1
            
    return accuracy, choices_probs_0 
