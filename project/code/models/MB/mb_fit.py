import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize

def mb_fit(df,num_of_parameters_to_recover):
    """
    this funcation performs parameters recovery of model based agent on the *reduce* two-stage task  
    Args:
        df: DataFrame of the behavior of the agent in the simulation
    Returns:
        res: results of the minimize funcation 
    """
    # sample initial guess of the parameters to recover 
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[1] = np.random.uniform(0.1,10)
    # set bounds to the recover parameters 
    bounds = [(0,1),(0.1,10)]
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
    state_of_stage_2 = list(map(int, df['state_of_stage_2']))
    reward_list = list(map(int, df['reward'])) 

    # set up paramters for recovary    
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
        
        # get true first action
        action = action_stage_1[t]
        choices_probs[t] = np.exp(beta*(q_mb[action])) / np.sum(np.exp(beta*(q_mb))) 
        
        # get true state and reward 
        state = state_of_stage_2[t] 
        reward = reward_list[t]

        p_e = reward - q_mf[state] 
        
        q_mf[state] = q_mf[state] + alpha*p_e
        
    eps = 1e-7
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss
