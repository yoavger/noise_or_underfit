import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize

def nws_fit(df,num_of_parameters_to_recover):
    """
    this funcation performs parameters recovery of noisy Win-stay-loss-shift agent on the *reduce* two-stage task  
    Args:
        df: DataFrame of the behavior of the agent in the simulation
    Returns:
        res: results of the minimize funcation 
    """
    # sample initial guess of the parameters to recover 
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    # set bounds to the recover parameters 
    bounds = [(0,1) for _ in range(num_of_parameters_to_recover)]

    res = minimize(
                    fun=parameters_recovary,
                    x0=initial_guess,
                    args=df,
                    bounds=bounds,
                    method='L-BFGS-B'
    )
    return res

def parameters_recovary(parameters, df):
	
    # objective to minimize
    log_loss = 0 

    num_of_trials = len(df)
    choices_probs = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    reward_list = list(map(int, df['reward'])) 

    # set up paramters for recovary    
    epsilon = parameters[0]

    for t in range(num_of_trials):
        if t == 0:
            p = np.array([.5,.5])
        else:
            if rLast == 1:  
                p = (epsilon/2)*np.array([1,1])
                p[aLast] = 1 - (epsilon/2)
            else:
                p = (1-(epsilon/2) )*np.array([1,1])
                p[aLast] = epsilon/2
                
        choices_probs[t] = p[action_stage_1[t]]
        aLast = action_stage_1[t]  
        rLast = reward_list[t]
        
    eps = 1e-7
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss