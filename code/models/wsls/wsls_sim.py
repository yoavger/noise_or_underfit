import numpy as np
import pandas as pd
from utils import *

def wsls_sim(parameters, num_of_trials, transition_prob, reward_probs):
    """
    this funcation simulate an Win-stay-loss-shift agent in the *reduce* two step task 
    Args:
        param: parameters of the agent 
        num_of_trials: number of trials of the simulation
        transition_prob: a matrix 2*2 of the transition function from the first stage to the second stage 
        reward_probs: a matrix num_of_trials*2*2 of the probability for reward of both second stage 
                      states for all trials
    
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
    p_stay_win = parameters['p_stay_win']
    p_shift_lose = parameters['p_shift_lose']
   
    state_transition_mat = np.array(
                            [[2,1],
                            [1,2]])
    # store data from each trial 
    data = DataOfSim(num_of_trials)

    for t in range(num_of_trials):
        if t == 0:
            # chose first action randomly 
            p = np.array([.5,.5])
        else:
            if rLast == 1:
                # win stay with probability p_stay_win
                p[aLast] = p_stay_win
                p[1-aLast] = 1 - p_stay_win
            else:
                # loss shift with probability p_shift_lose
                p[1-aLast] = p_shift_lose
                p[aLast] = 1 - p_shift_lose
        
        # choose an action according to action probabilities
        action = np.random.choice([0,1] , p=p)
                
        # sample a transation type
        transition_type = np.random.choice([0,1], p=transition_prob[1,:]) # 0 = rare / 1 = common
        # transation to second stage according to action and transation type
        state = state_transition_mat[action,transition_type]

        # check if the trial is rewarded
        reward = np.random.choice([0,1], p=reward_probs[t,state-1,:]) 
        
        # last trial infromation
        aLast = action  
        rLast = reward

        # stroe data of the trial
        data.n_trial[t] = t  
        data.action_1_list[t] = action
        data.stage_2_state[t] = state - 1 # store as 0 and 1 for analysis
        data.transition_list[t] = transition_type
        data.reward_list[t] = reward
        data.correct[t] = 0 if reward_probs[t,0,0] <= 0.2 else 1 
        data.probs_action_0[t] = p[0]

    df = pd.DataFrame(data.createDic())
    return df