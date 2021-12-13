import numpy as np
import pandas as pd
from utils import *

def kdh_predict(df,parameters):
    """
    this funcation predict the action of a recoverd k-Dominant Hand agent in the *reduce* two step task 
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
    choices_probs_0 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))

    # set up paramters 
    k = len(parameters)  
    p_dh = np.array( [parameters[i] for i in range(k)] )
    
    for t in range(num_of_trials):
        p = p_dh[t%k]
        p = [p, 1-p]

        # predict action according max probs 
        action_predict = np.argmax(p)
        choices_probs_0[t] = p[0]

        # cheek if prediction match the true action
        if action_predict == action_stage_1[t]:
            accuracy+=1
            
    return accuracy, choices_probs_0 
