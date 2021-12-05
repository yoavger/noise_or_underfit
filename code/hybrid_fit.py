import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import minimize


def hybrid_fit(df,num_of_parameters_to_recover):
    """
    this funcation performs parameters recovery of hybrid agent on the educe two-stage task  
    Args:
        df: DataFrame of the behavior of the agent in the simulation
    Returns:
        res: results of the minimize funcation 
    """
    # sample initial guess of the parameters to recover 
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[1] = np.random.uniform(0.1,10)
    # set bounds to the recover parameters 
    bounds = [(0,1),(0.1,10),(0,1)]
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
    p_choice_0 = np.zeros(num_of_trials)

    
    # upload data of the subject/agent
    action_stage_1 = list(map(int, df['action_stage_1']))
    state_of_stage_2 = list(map(int, df['state_of_stage_2']))
    reward_list = list(map(int, df['reward'])) 

    # set up paramters for recovary    
    alpha = parameters[0]
    beta = parameters[1]
    w = parameters[2]
    eligibility_trace = 0.9

    q_mf = np.zeros(shape=(3,2))
    q_mb = np.zeros(2)
    q_net = np.zeros(2)
    
    # state transition funcation 
    transition_prob = np.array([[.8,.2],
                                [.2,.8]])

    for t in range(num_of_trials):

        # Q_model-based values of the first stage actions - Bellman’s equation 
        q_mb[0] = (transition_prob[0,0]*q_mf[1,0]) + (transition_prob[0,1]*q_mf[2,0])
        q_mb[1] = (transition_prob[1,0]*q_mf[1,0]) + (transition_prob[1,1]*q_mf[2,0])

        # weighted sum of model-based and model-free action values
        q_net[0] = (w*q_mb[0]) + ((1-w)*q_mf[0,0])  
        q_net[1] = (w*q_mb[1]) + ((1-w)*q_mf[0,1])
        
        # get true first action
        action_1 = action_stage_1[t]
        p_choice_0[t] = np.exp(beta*(q_net[action_1])) / np.sum(np.exp(beta*(q_net))) 
        
        # get true state and reward 
        state = state_of_stage_2[t] + 1
        reward = reward_list[t]

        # calculate prediction error
        p_e_1 = q_mf[state,0] - q_mf[0,action_1] 
        p_e_2 = reward - q_mf[state,0] 
        
        # update q_mf according to q_learning formula
        q_mf[0,action_1] = q_mf[0,action_1] + alpha*p_e_1 + eligibility_trace*(alpha*p_e_2) 
        q_mf[state,0] = q_mf[state,0] + alpha*p_e_2
        
    eps = 1e-7
    log_loss = -(np.sum(np.log(p_choice_0 + eps)))
    return log_loss
