import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset , DataLoader

class behavior_dataset(Dataset):
    def __init__(self,data,length,if_state,if_state_one_back,if_trial):
        
        # action 
        action = np.array(data['action_stage_1'])
        if np.all(action == action[0]):
            action = np.append(action,(1-action[0]))
            action = torch.tensor((action).reshape(length+1),dtype=int)
            # one hot encoding 
            action_onehot = nn.functional.one_hot(action,len(action.unique()))
            # delete last one
            action_onehot = action_onehot[:-1]
        else:
            # action 
            action = torch.tensor((action).reshape(length),dtype=int)
            # one hot encoding 
            action_onehot = nn.functional.one_hot(action,len(action.unique()))
        
        # reward
        reward = torch.tensor((np.array(data['reward'])).reshape(length),dtype=int)
        
        # concatinating reward and action
        reward_action = torch.cat([reward[ :, np.newaxis], action_onehot],1)
        
        # adding dummy zeros to the beginning and ignoring the last one
        # [r (t-1) , a (t-1)]
        reward_action_shift = nn.functional.pad(reward_action,[0,0,1,0])[:-1]
        X = reward_action_shift
        y = action_onehot
        if if_state:
            state = torch.tensor((np.array(data['state_of_stage_2'])).reshape(length),dtype=int)
            # one hot encoding 
            state_onehot = nn.functional.one_hot(state,len(state.unique()))
            if if_state_one_back:
                state_onehot_shift = nn.functional.pad(state_onehot,[0,0,1,0])[:-1]
                # [r (t-1) , a (t-1) , s (t-1)]
                reward_action_state = torch.cat([reward_action_shift, state_onehot_shift],1)
            else:
                # [r (t-1) , a (t-1) , s (t)]
                reward_action_state = torch.cat([reward_action_shift, state_onehot],1)
   
            X = reward_action_state
        
        if if_trial:
            nTrial = torch.tensor((np.array(data['n_trial'])).reshape(len(data)),dtype=int)
            nTrial = nTrial.reshape(len(data),1)  
       
            # [r (t-1) , a (t-1) , s (t-1) , n_Trial]
            reward_action_state_nTrial = torch.cat([reward_action_state, nTrial],1)
            X = reward_action_state_nTrial
            
        self.x = X.type(dtype=torch.float32)
        self.y = action_onehot.type(dtype=torch.float32)
        
        self.len = length

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len    

class GRU_NN(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.hidden_size = hidden_size
        
        super(GRU_NN, self).__init__()
        self.hidden = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, hn = self.hidden(x)
        output = self.out(output)
        output = F.softmax(output,dim=-1)
        return output, hn

def train_model(net, train_loader, device, lr, epochs, input_size, output_size):
    """Simple helper function to train the model.
    
    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
    
    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    running_loss = 0
    r_l = 0
    
    # move to GPU
    net.to(device)
    # Loop over training batches
    for i in range(epochs):
        for j,(X_train,y_train) in enumerate(train_loader):
            
            # move to GPU
            X_train , y_train = X_train.to(device), y_train.to(device)
            
            # reshape to 1 X batch_size X input_size
            X_train = X_train.reshape(1,X_train.shape[0],input_size)
            
            # zero the gradient buffers
            optimizer.zero_grad() 
            out, hn = net(X_train)
            # Reshape to (SeqLen x Batch, OutputSize)
            out = out.view(-1, output_size)
       
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step() # Does the update
            
        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            r_l =  running_loss/100
            running_loss = 0 

    return net,r_l



def eval_network(rnn, test_data, input_size):
    nlp = 0
    acc = 0 
    ll = 0
    # eval
    X_test = test_data[0:][0]
    X_test = X_test.reshape(1,len(X_test),input_size)
    y_true = test_data[0:][1]

    
    with torch.no_grad():                  
        rnn.to('cpu')
        out = rnn(X_test)
        y_pred = out[0][0,0:] # probs of preidect action
        
        # nlp
        criterion = nn.BCELoss()
        cur_loss = criterion(y_pred, y_true)
        cur_loss = float(cur_loss.to('cpu').detach())
        nlp += cur_loss
        
        # acc
        y_pred_hard = torch.argmax(y_pred,1)
        y_true = torch.argmax(y_true,1)
        acc = y_pred_hard.eq(y_true).sum()/len(y_true)
        acc = float(acc.to('cpu').detach())

        # probs
#         ll = y_pred.gather(1, y_true.view(-1,1))
#         ll = torch.sum(torch.log(ll))
#         ll = torch.mean(ll)
#         ll = float(ll.to('cpu').detach())

    return nlp, acc
