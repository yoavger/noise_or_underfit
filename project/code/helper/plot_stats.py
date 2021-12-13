import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 

def update_data_frame(df): 
    # this funcation add to the data frame 3 columns:
    # 1 - prev_reward : if the last trail was rewarded
    # 2 - transation_prev : if the last transation was rare or common
    # 3 - stay probs 
    df['prev_reward'] = df['reward'].shift(1,fill_value=0)
    df['transition_prev'] = df['transition_type'].shift(1,fill_value=0)
    df['stay'] = df['action_stage_1'].shift(1)==df['action_stage_1']
    
def df_stay_probs(df): 
    # this funcation retuen a new df of stay prob according to every 
    # combination of reward/transation 4 option
    return df.groupby(['prev_reward', 'transition_prev'])['stay'].mean().reset_index()

def df_stay_to_list(df):
    # covert the data frame of stays to a list 
    df_stay = df_stay_probs(df)
    return df_stay['stay'].tolist()
    
def plot_stay_probs(df):
    # this funcation plots the stay probs given a list of the stay probs
    # each element of the list represents combination of transation/reward
    # list_of_stay[0] -> Unrewarded/Rare | list_of_stay [1] -> Unrewarded/Common
    # list_of_stay [2] -> Rewarded/Rare | list_of_stay[3] -> Rewarded/Common 
    df_stay = df_stay_probs(df)
    list_of_stay = df_stay_to_list(df_stay)
    
    x_labels = ['Rewarded' ,'Unrewarded']
        
    common = [list_of_stay[3],list_of_stay[1]]
    rare = [list_of_stay[2],list_of_stay[0]]
    
    # the width of the bars
    widthB = 0.35  
    r1 = np.arange(len(x_labels))
    r2 = [i + widthB for i in r1]
    
    fig, ax = plt.subplots()

    ax.bar(r1,common, color='blue', width=widthB, edgecolor='white', label='Common')
    ax.bar(r2, rare, color='red', width=widthB, edgecolor='white', label='Rare')

    ax.set_ylabel('Stay Probability',size=12)
    ax.set_xticks([((2*r + widthB)/2) for r in range(len(x_labels))])
    ax.set_xticklabels(x_labels,size=12)
    ax.set_ylim((0, 1))  
    ax.legend()
    
    fig.suptitle("Stay probs" , size=14)
    fig.tight_layout()
    plt.show()
    
    return ax
     
def calc_main_effect(df):
     # this funcation retuen the main effect of stay prob
    df_stay = df_stay_probs(df)
    list_of_stay = df_stay_to_list(df_stay)
    
    return ((list_of_stay[3]+list_of_stay[2])/2) - ((list_of_stay[1]+list_of_stay[0])/2)

def calc_interaction_effect(df):
    # this funcation retuen the interaction effect of stay prob
    df_stay = df_stay_probs(df)
    list_of_stay = df_stay_to_list(df_stay)
    return (list_of_stay[3] - list_of_stay[2]) - (list_of_stay[1] - list_of_stay[0])



