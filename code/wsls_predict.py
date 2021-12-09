import numpy as np
import pandas as pd
from utils import *

def wsls_predict(parameters, df):
    """
    this funcation predict the action of a recoverd wsls agent in the *reduce* two step task 
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
    reward_list = list(map(int, df['reward'])) 


    # state transition funcation 
    transition_prob = np.array(
        [[.8,.2],
        [.2,.8]]
    ) 

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    reward_list = list(map(int, df['reward'])) 

    # set up paramters for recovary    
    p_stay_win = parameters[0]
    p_stay_loss = parameters[1]
    
    p = np.array([.5,.5])

    for t in range(num_of_trials):
        if t > 0:
            if rLast == 1:
                p[aLast] = p_stay_win
                p[1-aLast] = 1 - p_stay_win
            else:
                p[aLast] = p_stay_loss
                p[1-aLast] = 1 - p_stay_loss
                
        action_1_predict = np.argmax(p)
        p_choice_0[t] = p[0]
        aLast = action_stage_1[t]  
        rLast = reward_list[t]

        # cheek if prediction match the true action
        if action_1_predict == aLast:
            accuracy+=1
            
    return accuracy, p_choice_0 
