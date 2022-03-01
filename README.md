# noise (working_title)
git repo accompanying the project 

## Background 
Theory-driven computational modeling allows estimation of latent cognitive and neural variables.  
Nonetheless, studies have found that computational models can systematically fail to describe some individuals.  
Presumably two factors contribute to this shortcoming:  
1. Higher internal noise (stochasticity in behavior).
2. Deployment of different model (unknown to the researcher).  

However, when measuring behavior of individuals on cognitive task, these two factors are entangled and therefore hard to dissociate.  
Here we examine (in-silico) the use of advances in data-driven machine learning algorithms disentangled this two factors.  

## Dataset
The dataset inclueded behavior of 500 artificial agent of 5 theoretical models (100 agent from each model):
1. Model-Free RL (MF). 
2. Model-Based RL (MB). 
3. Habitual model (habit). 
4. Win-Stay, Lose-Shift (WSLS). 
5. k-Dominant hand (kdh).  

Each agent was simulated on the reduce Two-Step task (TST) for 5 blocks containing 200 trials each. 
Post simulation we pretended that the true generative theoretical model of each agent is unknown and sought to ask if Recurrent neural network (RNN) and Logistic regression (LR), two models we considered as theory-independent can help classify each agent underlying theoretical model. 

## Leave-one-block out cross-validation (LOOCV)
To test this we act as follow. At each round (5 in total) we assumed that all agents came from only one theoretical model and compared the fit (predictive accuracy) of the assumed theoretical model against the fit of RNN and LR. We adapted an Leave-one-block out cross-validation approach. For each agent 4 block of his behavior (800 trials) were used to train a model and 1 witheld block (200 trials) was used to see how well the fitted model can generalize to unseen data. We avarged acrross all witheld blocks to obtain a single predictive accuracy score we denote as **nlp_m^i** (negative log probability; lower is better; m for the model used to fit behavior; i for the agent index). Calculating the difference between the **nlp_m^i** of the assumed theoretical model and the two theory-independent models allow us to say which model better explains the behavior of each agent. (lower **nlp** better explantion). We classified each agent to one of two categories, assumed theoretical model or unknown model. This can also be formultaed as a one-vs-all classifaction probelm. 

## ROC curve. Classifaction of generative model for each round (different assumed model).   
![image](https://github.com/yoavger/noise_or_underfit/blob/main/plots/roc_0.png)
We used RNN in three condition for classification. In two condition we fixed the number of training iteration for all agent (100 and 1000). In the third condition we varied the number of training iteration for each agent (early stopping). Best performance is acchived in the third condition, suppressing the fix condition and LR. 
all agents behavior can be found in the data f
```
noise_or_underfit/code/analysis/classification_roc.ipynb
```

## Difference in the averaged **nlp** 
![image](https://github.com/yoavger/noise_or_underfit/blob/main/plots/bar_plot.png)
Here, for each group of agents (5 groups corresponding to the 5 theoretical models) we averaged the **nlp** that each model obtains. Then we calculated the diffrence between the mean  **nlp** of the true theoretical model and the mean **nlp**  of all other models.  Across all models RNN in second only for the true generative model of each group of agents.
```
noise_or_underfit/code/analysis/nlp_bar_plot.ipynb
```

## Running the experiments 
- simulating the agents, fitting the 5 theoretical models and Logistic regression model and calculating the **nlp** on withheld blocks
run the following notebook:
```
noise_or_underfit/code/sim_and_fit.ipynb
```
the notebook saves each agent behavior in a csv file ```noise_or_underfit/data``` file in the following format:```{model}_agent_{# agent}_sim_{# block}```
and also creeat a csv file under ```noise_or_underfit/results``` with the results of the fitting and predecation of each mdoel. 

- fitting the rnn model: 
```
noise_or_underfit/code/rnn_fit.ipynb
```

- diffrent analysis on the data:
```
noise_or_underfit/code/analysis/
```
