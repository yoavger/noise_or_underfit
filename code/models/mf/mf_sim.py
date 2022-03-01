import numpy as np
import pandas as pd
from utils import *

def mf_sim(parameters, num_of_trials, transition_prob, reward_probs):
    """
    this funcation simulate an model-free agent in the *reduce* two step task 
    Args:
        param: parameters of the agent 
        num_of_trials: number of trials of the simulation
        transition_prob: a matrix 2*2 of the transition function from the first stage to the second stage 
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
    alpha = parameters['alpha']
    beta = parameters['beta']
    eligibility_trace = 0.9 # keep constant to all agent  

    # initialize Q-values
    # index 0 state / index 1 action
    q_mf = np.zeros(shape=(3,2))

    state_transition_mat = np.array([[2,1],
                                    [1,2]])
    # store data from each trial 
    data = DataOfSim(num_of_trials)

    for t in range(num_of_trials):
        
        # calculate probability for choosing action 0 in the for first stage with softmax choice rule 
        p = np.exp( beta*(q_mf[0,:])) / np.sum( np.exp(beta*q_mf[0,:]) ) 

        # choose action according to the probability of the first stage
        action = np.random.choice([0,1] , p=p)

        # sample a transation type
        transition_type = np.random.choice([0,1], p=transition_prob[1,:]) # 0 = rare / 1 = common
        # transation to second stage according to action and transation type
        state = state_transition_mat[action,transition_type]

        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=reward_probs[t,state-1,:]) 

        # calculate prediction error
        p_e_1 = q_mf[state,0] - q_mf[0,action] 
        p_e_2 = reward - q_mf[state,0] 

        # update q_mf according to q_learning formula
        q_mf[0,action] = q_mf[0,action] + alpha*p_e_1 + eligibility_trace*(alpha*p_e_2) 
        q_mf[state,0] = q_mf[state,0] + alpha*p_e_2
        
        # stroe data of the trial
        data.n_trial[t] = t  
        data.action_1_list[t] = action
        data.stage_2_state[t] = state - 1 # store as 0 and 1 for analysis
        data.transition_list[t] = transition_type
        data.reward_list[t] = reward
        data.correct[t] = 0 if reward_probs[t,0,0] <= 0.2 else 1 
        data.probs_action_0[t] = p[0]
        data.delta_q[t] = q_mf[0,0] - q_mf[0,1]
       
    df = pd.DataFrame(data.createDic())
    return df
