import numpy as np
import random
import time
import os
import models
import math
import torch.nn.functional as F
import torch



"""
Contains the definition of the agent that will run in an
environment.
"""
def epsilon(epsilon_0, A, B, C, time, EPISODES):
    standardized_time=(time-A*EPISODES)/(B*EPISODES)
    cosh=np.cosh(math.exp(-standardized_time))
    epsilon=epsilon_0-(1/cosh+(time*C/EPISODES))
    return epsilon



class DQAgent:

    def __init__(self,graph,lr,phrase):

        self.graphs = graph   
        self.phrase = phrase

            



    def reset(self, g , epoch_, graph_num):
        
        self.epoch_ = epoch_
        self.games = graph_num
        self.graph_num = graph_num           
        if g == 1:
            self.mode = 'train'
        elif g == 2:
            self.mode = 'RLtest'
        else:
            self.mode = 'RDtest'


    def act(self, observation):
        return np.random.choice(np.where(observation.numpy()[0,:,0] != 0)[0])   

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd+'/model_1200.pt')
