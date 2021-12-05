import numpy as np
import pandas as pd
from utils import *

def hybrid_predict(paramters, df):
    """
    this funcation predict the action of a recoverd hybrid agent in the *reduce* two step task 
    Args:
        paramters: parameters of the agent 
        df: DataFrame of the true behavior of the agent in the simulation we want to predict 
        
    
    Returns:
    accuracy - number of action predicted correctly (argmax) 
    p_choice_0 - a vector of length num_of_trials of the probability of choosing action 0 in the first stage
       
        
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
    p_choice_0 = np.zeros(num_of_trials)
 

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    state_of_stage_2 = list(map(int, df['state_of_stage_2']))
    reward_list = list(map(int, df['reward'])) 

    # set up paramters of the agent     
    alpha = paramters[0]
    beta = paramters[1]
    w =  paramters[2]
    eligibility_trace = 0.9

    # set up paramters of the hybrid agent 
    q_mf = np.zeros(shape=(3,2))
    q_mb = np.zeros(2)
    q_net = np.zeros(2)

    # state transition funcation 
    transition_prob = np.array(
        [[.8,.2],
        [.2,.8]]
    ) 

    for t in range(num_of_trials):

        # Q_model-based values of the first stage actions - Bellman’s equation 
        q_mb[0] = (transition_prob[0,0]*q_mf[1,0]) + (transition_prob[0,1]*q_mf[2,0])
        q_mb[1] = (transition_prob[1,0]*q_mf[1,0]) + (transition_prob[1,1]*q_mf[2,0])

        # weighted sum of model-based and model-free action values
        q_net[0] = (w*q_mb[0]) + ((1-w)*q_mf[0,0])  
        q_net[1] = (w*q_mb[1]) + ((1-w)*q_mf[0,1])

        # calculate probability for choosing action 0 in the for first stage with softmax choice rule 
        prob_0 = np.exp( beta*(q_net[0]) ) / np.sum( np.exp( beta*(q_net))) 
        p_choice_0[t] = prob_0
        
        # choose action_1 according max probs 
        action_1_predict = np.argmax([prob_0, 1-prob_0])

        # get true action, state, and reward 
        action_1_true = action_stage_1[t]
        state = state_of_stage_2[t] + 1
        reward = reward_list[t]
    
        # calculate prediction error
        p_e_1 = q_mf[state,0] - q_mf[0,action_1_true] 
        p_e_2 = reward - q_mf[state,0] 

        # update q_mf according to q_learning formula
        q_mf[0,action_1_true] = q_mf[0,action_1_true] + alpha*p_e_1 + eligibility_trace*(alpha*p_e_2) 
        q_mf[state,0] = q_mf[state,0] + alpha*p_e_2

        # cheek if prediction match the true action
        if action_1_predict == action_1_true:
            accuracy+=1
            
    return accuracy, p_choice_0 
