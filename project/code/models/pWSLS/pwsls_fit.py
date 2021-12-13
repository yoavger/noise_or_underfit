import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize

def pwsls_fit(df,num_of_parameters_to_recover):
    """
    this funcation performs parameters recovery of Probabilistic Win-stay-loss-shift agent on the educe two-stage task  
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
    p_stay_win = parameters[0]
    p_stay_loss = parameters[1]

    for t in range(num_of_trials):
        if t == 0:
            p = np.array([.5,.5])
        else:
            if rLast == 1:
                p[aLast] = p_stay_win
                p[1-aLast] = 1 - p_stay_win
            else:
                p[aLast] = p_stay_loss
                p[1-aLast] = 1 - p_stay_loss

        choices_probs[t] = p[action_stage_1[t]]
        aLast = action_stage_1[t]  
        rLast = reward_list[t]
        
    eps = 1e-7
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss