import numpy as np
import pandas as pd
from plot_stats import *

import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate ,ZeroPadding1D
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as kb
from tensorflow.python.keras.losses import categorical_crossentropy, binary_crossentropy

def preprocessing_from_csv(root_path, num_of_trials, arr):  
    """
    this funcation preprocessing the data from csv
    Args:
        root_path: root to read csv formated ...._sim_#.csv
        num_of_trials: number of trials of the simulation
        arr: an array of the blocks number to preproces


    Returns:
        reward: numpay array 1*(num_of_trials*num_blocks) of the reward given
        action: numpay array 1*(num_of_trials*num_blocks) of the action taken
        state: numpay array 1*(num_of_trials*num_blocks) of the state the agent visted
    """
    num_blocks = len(arr)
    data = np.zeros(shape=(num_of_trials*num_blocks,3))
    for i,sim in enumerate(arr):
        path = f'{root_path}_sim_{sim}.csv'
        df = pd.read_csv(path)
        for row in df.itertuples(index=True, name='Pandas'):    
            data[row.Index + num_of_trials*i] = row.reward, row.action_stage_1, row.state_of_stage_2 

    reward = (data[0:num_of_trials*num_blocks ,0]).reshape(1,num_of_trials*num_blocks)
    action = (data[0:num_of_trials*num_blocks ,1]).reshape(1,num_of_trials*num_blocks)
    state = (data[0:num_of_trials*num_blocks,2]).reshape(1,num_of_trials*num_blocks)
    
    return reward, action, state

def preprocessing_from_dataframe(df,num_of_trials,arr):
    """
    this funcation preprocessing the data from dataframe
    Args:
        df: df of the behavior 
        num_of_trials: number of trials of the simulation
        arr: an array of the blocks number to preproces


    Returns:
        reward: numpay array 1*(num_of_trials*num_blocks) of the reward given
        action: numpay array 1*(num_of_trials*num_blocks) of the action taken
        state: numpay array 1*(num_of_trials*num_blocks) of the state the agent visted
    """
   
    num_blocks = len(arr)
    data = np.zeros(shape=(num_of_trials*num_blocks,3))
    for row in df.itertuples(index=True, name='Pandas'):    
        data[row.Index] = row.reward, row.action_stage_1, row.state_of_stage_2 

    reward = (data[0:num_of_trials*num_blocks ,0]).reshape(1,num_of_trials*num_blocks)
    action = (data[0:num_of_trials*num_blocks ,1]).reshape(1,num_of_trials*num_blocks)
    state = (data[0:num_of_trials*num_blocks,2]).reshape(1,num_of_trials*num_blocks)
    
    return reward, action, state

def one_hot_encoding(reward, action, state, n_actions, n_state):
    """
    this funcation convert data to one hot encoding - 
    we are interested in training an rnn to predict the current action given the last trial information      
    Args:
        reward: numpay array 1*(num_of_trials*num_blocks) of the reward given
        action: numpay array 1*(num_of_trials*num_blocks) of the action taken
        state: numpay array 1*(num_of_trials*num_blocks) of the state the agent visted
        n_actions: number of action in the env
        n_state: number of state in the env

    Returns:
        reward_action_state: a tensor ( 1*(num_of_trials*num_blocks)*(1+n_action+n_state) ) 
                                        Concatenating rewared actoin of the last timestep and cuurent timestep state
        action_onehot: a tensor (1*(num_of_trials*num_blocks)*(n_action)) one hot vector of the action taken in the cuurent timestep
        
    """

    # action 0 is coded as [1 , 0]; and action 1 is coded as [0, 1]
    action_onehot = tf.one_hot(action, n_actions)

    # second stage (state 0) is coded as  [1, 0]
    # second stage (state 1) is coded [0, 1] 
    state_onehot = tf.one_hot(state, n_state)

    # concatinating reward and action to feed the RNN as input
    reward_action = Concatenate(axis=2)([reward[:, :, np.newaxis], action_onehot])

    # adding dummy zeros to the beginning and ignoring the last one
    reward_action = ZeroPadding1D(padding=[1, 0])(reward_action)[:, :-1, :]

    # adding dummy zeros to the beginning and ignoring the last one
    state_onehot_shift = ZeroPadding1D(padding=[1, 0])(state_onehot)[:, :-1, :]

    # [r (t-1) , a (t-1) , s (t-1)]
    reward_action_state = Concatenate(axis=2)([reward_action, state_onehot_shift])
    
    return reward_action_state, action_onehot

def create_model_gru(n_actions, n_state, n_cells):
    """
    this funcation define an RNN architecture based on GRU unit 
    input (lasst trial info + last hidden state) -> hidden (recurrent unit) -> output (probability of taking each action) - 
    Args:
        n_actions: number of action in the env
        state_size:  number of states in the env 
        n_cells: number of recurrent units in the hidden layer of the network 

    Returns:
        model: model of the RNN pre-train
        sim_model: model of RNN for off-policy simulations
    """
    # last trial information
    model_inputs = tf.keras.Input(shape=(None, 1 + n_actions + n_state), name='reward_action_state')

    # initial state of the RNN, Dim: S x n_cells
    rnn_initial_state = tf.keras.Input(shape=(n_cells,), name='initial_state')

    # output of the recurrent nerual network hiddent layer. Dim: S x T x n_cells
    rnn_out = kl.GRU(n_cells, return_sequences=True, name='GRU')(model_inputs, initial_state=rnn_initial_state)

    # policy -- probablity of taking each action. Dim: S x T x n_actions
    policy = kl.Dense(n_actions, activation='softmax', name='policy')(rnn_out)

    # defining the model
    model = tf.keras.Model(inputs=[model_inputs, rnn_initial_state], outputs=policy, name='model')

    # this model will be used for off-policy simulations in which we need to track rnn state
    sim_model = tf.keras.Model(inputs=[model_inputs,rnn_initial_state], outputs=[rnn_out, policy])

    return model, sim_model


def create_model_lstm(n_actions,n_state,n_cells):
    """
    this funcation define an RNN architecture based on LSTM unit 
    input (lasst trial info + last hidden state) -> hidden (recurrent unit) -> output (probability of taking each action) - 
    Args:
        n_actions: number of action in the env
        n_state:  number of states in the env 
        n_cells: number of recurrent units in the hidden layer of the network 

    Returns:
        model: model of the RNN pre-train
        sim_model: model of RNN for off-policy simulations

    """
    model_inputs = tf.keras.Input(shape=(None, 1 + n_actions + n_state), name='reward_action_state')

    # initial state of the RNN, Dim: S x n_cells
    rnn_initial_state = tf.keras.Input(shape=(n_cells,), name='initial_state')

    # output of the recurrent nerual network. Dim: S x T x n_cells
    rnn_out = kl.LSTM(n_cells, return_sequences=True, name='LSTM')(model_inputs,initial_state=[rnn_initial_state,rnn_initial_state])

    # policy -- probablity of taking each action. Dim: S x T x n_actions
    policy = kl.Dense(n_actions, activation='softmax', name='policy')(rnn_out)

    # defining the model
    model = tf.keras.Model(inputs=[model_inputs, rnn_initial_state], outputs=policy, name='model')

    # this model will be used for off-policy simulations in which we need to track rnn state
    sim_model = tf.keras.Model(inputs=[model_inputs,rnn_initial_state], outputs=[rnn_out, policy])

    return model, sim_model

def compile_model(model, lr, initial_state_size, n_cells):
    """
    this funcation compile an RNN model (define optimzer le etc)

    Args:
        model: model of the RNN pre-train
        lr:  learning rate of the optimizers
        initial_state_size: the size of the recurrent hidden layer of the network
        n_cells: number of recurrent units in the hidden layer of the network 

    Returns:
        model: model post compiltion
        initial_rnn_state: the initial rnn state (tensor of zeros)
    """
   
    # defining the optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # initial state of RNN 
    initial_rnn_state = np.zeros((initial_state_size, n_cells,), dtype=np.float32)

     # defining loss
    def loss(y_true, y_pred):
        return kb.sum(binary_crossentropy(y_true, y_pred))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy','BinaryCrossentropy'])

    return model, initial_rnn_state

def train_a_model_gru(train_data, train_arr,test_data, test_arr, num_of_trials, n_cells, lr, epochs, batch_size, cp_callback):
    """
    this funcation train an RNN model 

    Args:
        train_data: data to train on 
        train_arr:  block number of the train data 
        test_data: data to test on 
        test_arr: block number of the test data 
        num_of_trials: num of trials of each block 
        n_cells: number of recurrent units in the hidden layer of the network 
        lr:learning rate of the optimizers
        epochs: number of traning iteration 
        batch_size: number of trial to input the rnn each optimization step 
        cp_callback: root to save model weight 

    Returns:
        model: model post traning 
        sim_model: model for off-policy  simulations
        hist: accuarcy and loss of the train and test sets 
    """
    
    # preprocessing and transform to one hot encoding of the traing data
    reward, action, state = preprocessing_from_dataframe(
                                                          train_data,
                                                          num_of_trials,
                                                          train_arr
                                                        )

    X_train , y_train = one_hot_encoding(reward, action, state, n_actions=2, n_state=2)

    # preprocessing and transform to one hot  encoding of the test data
    reward, action, state = preprocessing_from_dataframe(test_data,
                                                          num_of_trials,
                                                          test_arr)

    X_test , y_test = one_hot_encoding(reward, action, state, n_actions=2, n_state=2)


    model, sim_model = create_model_gru(n_actions=2, n_state=2, n_cells=n_cells)

    model, initial_rnn_state = compile_model(   model,
                                                lr,X_train.shape[0],
                                                n_cells
                                            )

    # training the model
    hist = model.fit(x=[X_train, initial_rnn_state],
                     y=y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=0,
                     validation_data=([X_test,initial_rnn_state],y_test))
                    # callbacks=[cp_callback]

    return model, sim_model, hist



def on_policiy_tst(sim_model, num_of_trials, reversal_each_trials, n_cells):
    """
    this funcation simulates an RNN model on the reduce two-step task
    Args:
        sim_model: model to simulate on 
        num_of_trials:  num of trials of the task
        reversal_each_trials: num of trials to revers the better (in terms of rewared) states 
        n_cells: number of recurrent units in the hidden layer of the network 

    Returns:
        df: datafraem of the behavior of the rnn model 
        activity_dict: the activity of the recurrent units during the simulation
    """

    switch = False
    
    # initial actions and rewards
    action_onehot_play = np.zeros([1, 1, 2], dtype=np.float32)
    reward_play = np.zeros((1, 1, 1), dtype=np.float32)
    reward_action_play = Concatenate(axis=2)([reward_play, action_onehot_play])
    state_onehot_play = np.zeros([1, 1, 2], dtype=np.float32)
    reward_action_state_play = Concatenate(axis=2)([reward_action_play, state_onehot_play])

    # initial state of RNN
    rnn_state = np.zeros((reward_action_state_play.shape[0], n_cells,), dtype=np.float32)

    policies = np.zeros((num_of_trials,))
    actions = np.zeros((num_of_trials,))
    rewards = np.zeros((num_of_trials,))
    transations = np.zeros((num_of_trials,))
    # recording activity
    activity_dict = {}  

    # start simulation
    for t in range(policies.shape[0]):

        # getting the policy
        rnn_state, policy = sim_model([reward_action_state_play, rnn_state])
        policies[t] = np.squeeze(policy)[1]

        # gettig the last state of RNN
        rnn_state = rnn_state[:, 0, :]
        rnn_activity = rnn_state

        action_onehot = np.zeros([1, 1, 2], dtype=np.float32)
        state_onehot = np.zeros([1, 1, 2], dtype=np.float32)

        # sampling the action and transation 
        if np.random.random() < np.squeeze(policy)[0]:
            action_onehot[0, 0, 0] = 1
            actions[t] = 0
            if np.random.random() < 0.8:
                state_onehot[0,0,0] = 1
                transations[t] = 1
            else:
                state_onehot[0,0,1] = 1
                transations[t] = 0
        else:
            action_onehot[0, 0, 1] = 1
            actions[t] = 1
            if np.random.random() < 0.8:
                state_onehot[0,0,1] = 1
                transations[t] = 1 
            else:
                state_onehot[0,0,0] = 1
                transations[t] = 0

        # switch rewards probs   
        if t%reversal_each_trials == 0:
            if switch:
                switch = False
            else:
                switch = True

        # sampling rewards condtion on the state 
        if switch:
            if state_onehot[0,0,0] == 1:
                if np.random.random() < 0.8:
                    reward = np.ones((1, 1, 1), dtype=np.float32)
                    rewards[t] = 1
                else:
                    reward = np.zeros((1, 1, 1), dtype=np.float32)
                    rewards[t] = 0
            else:
                if np.random.random() < 0.2:
                    reward = np.ones((1, 1, 1), dtype=np.float32)
                    rewards[t] = 1
                else:
                    reward = np.zeros((1, 1, 1), dtype=np.float32)
                    rewards[t] = 0
        else:
            if state_onehot[0,0,0] == 0:
                if np.random.random() < 0.8:
                    reward = np.ones((1, 1, 1), dtype=np.float32)
                    rewards[t] = 1
                else:
                    reward = np.zeros((1, 1, 1), dtype=np.float32)
                    rewards[t] = 0
            else:
                if np.random.random() < 0.2:
                    reward = np.ones((1, 1, 1), dtype=np.float32)
                    rewards[t] = 1
                else:
                    reward = np.zeros((1, 1, 1), dtype=np.float32)
                    rewards[t] = 0


        activity_dict[t] = rnn_activity
        reward_action_play = Concatenate(axis=2)([reward, action_onehot])
        reward_action_state_play = Concatenate(axis=2)([reward_action_play, state_onehot])

    # plt.ylim((0,1))
    # plt.xlabel('tiral')
    # plt.ylabel('probablity of action 1')
    # plt.plot(np.arange(policies.shape[0]), policies)
    # plt.axhline(y=0.5, linewidth=2, color = 'red', linestyle="dashed")

    # store data 
    dic = {
    'action_stage_1' : actions,
    'transition_type' : transations,
    'reward' : rewards
    }
    
    df = pd.DataFrame(dic)
    update_data_frame(df)   
    return df, activity_dict
