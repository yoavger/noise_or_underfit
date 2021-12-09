import numpy as np
import pandas as pd
from utils import *

def wsls_sim(param, num_of_trials, transition_prob, reward_probs):
    """
    this funcation simulate an win-stay-loss-shift agent in the *reduce* two step task 
    Args:
        param: parameters of the agent 
        num_of_trials: number of trials of the simulation
        transition_prob: a matrix 2*2 of the transition function front the first stage to the second stage 
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
    p_stay_win = param['p_stay_win']
    p_stay_loss = param['p_stay_loss']
   
    state_transition_mat = np.array(
                            [[2,1],
                            [1,2]]
                        )
    
    # chose first action randomly 
    p = np.array([.5,.5])
    
    # store data from each trial 
    data = DataOfSim(num_of_trials)

    for t in range(num_of_trials):
        if t > 0:
            if rLast == 1:
                # win stay with probability p_stay_win
                p[aLast] = p_stay_win
                p[1-aLast] = 1 - p_stay_win
            else:
                # loss stay with probability p_stay_loss
                p[aLast] = p_stay_loss
                p[1-aLast] = 1 - p_stay_loss
        
        # choose an action according to action probabilities
        action = np.random.choice([0,1] , p=p)
                
        # sample a transation type
        transition_type = np.random.choice([0,1], p=transition_prob[1,:]) # 0 = rare / 1 = common
        
        # transation to second stage according to action and transation type
        state = state_transition_mat[action,transition_type]

        # check if the trial is rewarded
        reward = np.random.choice([0,1], p=reward_probs[t,state-1,:]) 
        
        # store last trial infromation
        aLast = action  
        rLast = reward
        
        correct = 0 if reward_probs[t,0,0] <= 0.2 else 1 
            
        # stroe data of the trial
        data.action_1_list[t] = action
        data.stage_2_state[t] = state - 1 # store as 0 and 1 for easier analysis
        data.transition_list[t] = transition_type
        data.reward_list[t] = reward
        data.correct[t] = correct
        data.probs[t] = p[0]

    df = pd.DataFrame(data.createDic())
    return df