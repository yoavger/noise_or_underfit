import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize

def habit_fit(df,num_of_parameters_to_recover):
    """
    this funcation performs parameters recovery of habitual agent on the *reduce* two-stage task  
    Args:
        df: DataFrame of the behavior of the agent in the simulation
    Returns:
        res: results of the minimize funcation 
    """
    # sample initial guess of the parameters to recover 
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[1] = np.random.uniform(0.1,4)
    # set bounds to the recover parameters 
    bounds = [(0,1),(0.1,4)]
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

    # set up paramters for recovary    
    alpha = parameters[0]
    beta = parameters[1]
    
    # initialize Habit matrix
    H = np.zeros(2) + .5

    for t in range(num_of_trials):  
        
        # calc prob with softmax for first stage
        p = np.exp(beta*(H)) / np.sum(np.exp(beta*H)) 

        # get true first action
        action = action_stage_1[t]
        choices_probs[t] = p[action]
        
        H = (1-alpha)*H
        H[action] = H[action] + alpha
  
    eps = 1e-7
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss


