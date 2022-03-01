import pandas as pd 
import numpy as np
from plot_stats import * 
from sklearn.linear_model import LogisticRegression
import torch 
import torch.nn as nn


def preprocess_logistic_regression(data):
    """
    this funcation preprocessing the data for logistic regression 
    we are interested in predicting the stay probs of repeating the last action 
    using th last transition type, last reward and the interaction of the two as regressors
    Args:
        data: dataframe of behavior to fit  

    Returns:
        X: a matrix of the predictors variables
        y: a vector of the response variable 
        
    """
    update_data_frame(data)
    df = data
    
    df['stay'] = df['stay'].astype(int)

    # center the data
    df.loc[ (df['transition_prev'] == 0) & (df['prev_reward'] == 1),'interaction'] = -0.5
    df.loc[ (df['transition_prev'] == 1) & (df['prev_reward'] == 0),'interaction'] = -0.5

    df.loc[ (df['transition_prev'] == 0) & (df['prev_reward'] == 0),'interaction'] = 0.5
    df.loc[ (df['transition_prev'] == 1) & (df['prev_reward'] == 1),'interaction'] = 0.5

    df.loc[df['prev_reward'] == 0,'prev_reward'] = -0.5
    df.loc[df['prev_reward'] == 1,'prev_reward'] = 0.5

    df.loc[df['transition_prev'] == 0,'transition_prev'] = -0.5
    df.loc[df['transition_prev'] == 1,'transition_prev'] = 0.5

    y = df['stay'][1:]
    y = np.array(y)
    if np.all(y == y[0]):
        y[0] = (1-y[0])
    X = np.vstack(( df['prev_reward'][1:],
                    df['transition_prev'][1:],
                    df['interaction'][1:])
    )
    # transpose the design matrix
    X = X.T

    return X, y

def fit_logistic_regression(X,y):
    """
    fit logistic regression on the data
    
    Args:
        X: a matrix of the predictors variables
        y: a vector of the response  

    Returns:
        lm: the model aftre fit
        lm.intercept_: the intercept coefficient
        lm.coef_ : the coefficient of the other predictors
    """
    lm = LogisticRegression(fit_intercept=True, solver='liblinear')
    lm.fit(X, y)
    return lm, lm.intercept_, lm.coef_ 

def nlp_logistic_regression(model,X,y):
    """
    calculate the negative log probability on a test data of a logistic regression model
    according to the binary cross entropy loss funcation:

        -( y_true * log(y_predict) + (1-y_true)*log(1 - y_predict) )
    
    Args:
        model: model to calculate on 
        X: a matrix of the predictors variables
        y: a vector of the response 

    Returns:
        running_loss: the total negative log probability (loss)
    """
    criterion = nn.BCELoss()
    p = model.predict_proba(X)
    running_loss = 0
    for label,probs in zip(y,p):
        y_pred = torch.tensor([1-probs[0]],dtype=torch.float32)
        y_true = torch.tensor([label],dtype=torch.float32)
        running_loss += (criterion(y_pred,y_true)).numpy()
    return running_loss


