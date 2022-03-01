import numpy as np
import pandas as pd

# utility funcation for configuration, simulation, storing the data  

def bce_loss(y_true ,y_predict):
    # binary cross entrophy loss with log base 2 
    return -( y_true * np.log2(y_predict) + (1-y_true)*np.log2(1 - y_predict) )

def create_reward_probs(num_of_trials, reversal, low, high):
    # creat rewards probabilities num_of_trials times
    # and reverse the low and high probs every reversal times 

    reward = np.zeros(shape=(num_of_trials,2,2))

    # randomley selcet the first best choice 
    p_1 = np.random.choice([low,high])
    reward_mat = np.array(
              [[p_1,1-p_1],
              [1-p_1,p_1]]
     )
    for t in range(num_of_trials):
        if t%reversal==0:
            # reverse
            reward_mat[[0, 1]] = reward_mat[[1, 0]] 
        reward[t] = reward_mat
    return reward

def configuration_parameters_mf():
    # 2 free parameters (α, β)     
    parameters = {
                'alpha' : np.random.uniform(), # 0 <= alpha <= 1 
                'beta' : np.random.uniform(0.1,10), # 0 <= beta <= 10  inverse temperature beta of stage 1  
    }
    return parameters

def configuration_parameters_mb():
    # 2 free parameters (α, β)     
    parameters = {
                'alpha' : np.random.uniform(), # 0 <= alpha <= 1 
                'beta' : np.random.uniform(0.1,10), # 0 <= beta <= 10  inverse temperature beta of stage 1  
    }
    return parameters

def configuration_parameters_wsls():
    # 2 free parameters p_stay_win, p_shift_lose     
    parameters = {
                'p_stay_win' : np.random.uniform(), # 0 <= p_stay_win <= 1
                'p_shift_lose' : np.random.uniform() # 0 <= p_shift_lose <= 1
    }
    return parameters


def configuration_parameters_habits():
    # 2 free parameters (α, β)     
    parameters = {
                'alpha' : np.random.uniform(), # 0 <= alpha <= 1 
                'beta' : np.random.uniform(0.1,4), # 0 <= beta <= 10  inverse temperature beta of stage 1  
    }
    return parameters

def configuration_parameters_kdh():
    # k free parameters p_0 , p_1 , ... , p_k
    k = 2     
    parameters = {
                'p_dh' : [np.random.uniform() for i in range(k)],
                'k' : k # k
    }
    return parameters

class DataOfSim():
    # this class stores all the data of one simulation
    # storing the following: action_1, stage_2_state, transation_type, action_2, reward

    def __init__ (self , num_of_trials):
        self.n_trial =  np.zeros(num_of_trials,dtype=int)
        self.action_1_list = np.zeros(num_of_trials,dtype=int)
        self.stage_2_state = np.zeros(num_of_trials,dtype=int)
        self.transition_list =  np.zeros(num_of_trials,dtype=int)
        self.reward_list = np.zeros(num_of_trials,dtype=int) 
        self.probs_action_0 = np.zeros(num_of_trials,dtype=np.float32)
        self.correct = np.zeros(num_of_trials,dtype=int)
        self.delta_q = np.zeros(num_of_trials,dtype=np.float32)

    def createDic(self):
        dic = {
                'n_trial':self.n_trial,
                'action_stage_1' : self.action_1_list,
                'state_of_stage_2' : self.stage_2_state,
                'transition_type' : self.transition_list,
                'reward' : self.reward_list,
                'probs_action_0' : self.probs_action_0,
                'correct': self.correct,
                'delta_q': self.delta_q       
            }
        return dic 
