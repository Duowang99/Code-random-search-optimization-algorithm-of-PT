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
        self.embed_dim = 4

        self.k = 20
        self.alpha = 0.1
        self.gamma = 0.95
        self.lambd = 0.
        self.n_step = 1
        
        
        #hyper-parameters for updating epsilon
        self.epsilon_0 = 1.1
        self.epsilon_=1.1
        self.A=0.5
        self.B=0.1
        self.C=0.1
        self.EPISODES = 6
        
        self.t=1
        
        #for memory
        self.memory = []
        self.memory_n = []
        self.minibatch_length = 1
        
        self.phrase = phrase
        if self.phrase == 'train':
            self.model = models.S2V_QN_1(64,64,0,0,5)
#             cwd = os.getcwd()
#             PATH = cwd+'/model_1000.pt'
#             print(PATH)
#             self.model.load_state_dict(torch.load(PATH))
#             self.model.eval()
        else:
            self.model = models.S2V_QN_1(64,64,0,0,5)
#             cwd = os.getcwd()
#             PATH = cwd+'/model_1200.pt'
#             print(PATH)
#             self.model = models.S2V_QN_1(64,64,0,0,5)
#             self.model.load_state_dict(torch.load(PATH))
#             self.model.eval()
            
            
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.T = 5
        self.q_change_rate_list = []


    def reset(self, g , epoch_, graph_num):
        
        self.epoch_ = epoch_
        self.games = graph_num
        self.graph_num = graph_num        
        
        
#         if (len(self.memory_n) != 0) and (len(self.memory_n) % 300000 == 0):
#             self.memory_n =random.sample(self.memory_n,120000)
        
        

        self.nodes = self.graphs[self.graph_num].nodes()
        self.adj = self.graphs[self.graph_num].adj()
        self.adj = self.adj.todense()
        self.adj = torch.from_numpy(np.expand_dims(self.adj.astype(int), axis=0))
        self.adj = self.adj.type(torch.FloatTensor)
        
        
        self.last_action = 0
        self.last_observation = torch.ones(1, self.nodes, 1, dtype=torch.float)
        self.last_reward = -0.01
        self.last_done=0
        self.iter=1

        
        if g == 1:
            self.mode = 'train'
        elif g == 2:
            self.mode = 'RLtest'
        else:
            self.mode = 'RDtest'
        self.epsilon_0 = 1.1
        self.epsilon_=1.1

    def act(self, observation):
        
        if self.mode == 'RDtest':
#            print('      *random agent*')
            return np.random.choice(np.where(observation.numpy()[0,:,0] != 0)[0])   
        elif self.epsilon_ > np.random.rand() and self.mode == 'train':
            print('      *random smaller than epsilon*')
            return np.random.choice(np.where(observation.numpy()[0,:,0] != 0)[0]) 
        else:
            print('      *not random bigger than epsilon*')
            input_ = observation
            q_a = self.model(input_, self.adj)
            q_a=q_a.detach().numpy().reshape(-1)
            
            obser = observation.numpy().reshape(-1)
            max_ = np.max( q_a[ obser != 0. ] )
            for i in range(len(obser)):
                if q_a[i] == max_ and obser[i] != 0.:
                    return i


    

    def reward(self, observation, action, reward,done):
#         if self.mode == 'train':
#             old_input = old_observation
#             new_input = new_observation

#             q_table_before_train = self.model( old_input, self.adj).detach().numpy().reshape(-1)

#             q_a = self.model( new_input, self.adj).detach().numpy()
#             max_ = np.max(   q_a[0, :, 0][new_observation.numpy()[0, :, 0] != 0] )
#             target = self.gamma * max_ + reward
# #             print('target',target)
#             target_f = self.model( old_input, self.adj)
#             target_p = target_f.clone()

#             target_f[0,action,0] = target
#             loss=self.criterion(target_p, target_f)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
# #             print('target_f',target_f)
# #             print('target_p',target_p)
#             q_table_after_train = self.model( old_input, self.adj).detach().numpy().reshape(-1)
    
#             q_change_rate = np.linalg.norm( q_table_after_train - q_table_before_train ) / np.linalg.norm( q_table_before_train )
             
#             print(  '    q_norm_change_rate: ',q_change_rate )
#             #update epsilon
# #             self.epsilon_ = epsilon( self.epsilon_0, self.A, self.B, self.C, self.iter, self.EPISODES )
#             print( '    epsilon: ',self.epsilon_ )
            
#             self.iter+=1
#             self.t += 1
#             self.epsilon_ = epsilon( self.epsilon_0, self.A, self.B, self.C, self.iter, self.EPISODES )
#             self.q_change_rate_list.append( q_change_rate )
            
        if self.mode == 'train':
            if len(self.memory_n) > self.minibatch_length + self.n_step: #or self.games > 2:

                (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens,adj_tens) = self.get_sample()
                target = reward_tens+self.gamma *(1-done_tens)*torch.max(self.model(observation_tens,adj_tens)+observation_tens*(-1e5),dim=1)[0]
                target_f = self.model(last_observation_tens, adj_tens)
                target_p = target_f.clone()
                target_f[range(self.minibatch_length),action_tens,:] = target
                loss=self.criterion(target_p, target_f)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print(self.t, loss)

            if self.iter>1:
                self.remember(self.last_observation, self.last_action, self.last_reward, observation.clone(),self.last_done*1)

            if done & self.iter> self.n_step:
                self.remember_n(False)
                new_observation = observation.clone()
                new_observation[:,action,:]=1
                self.remember(observation,action,reward,new_observation,done*1)

            if self.iter > self.n_step:
                self.remember_n(done)
            self.iter+=1
            self.t += 1
            self.last_action = action
            self.last_observation = observation.clone()
            self.last_reward = reward
            self.last_done = done

            
    def get_sample(self):
        
        minibatch = random.sample(self.memory_n, self.minibatch_length - 1)
        minibatch.append(self.memory_n[-1])
        last_observation_tens = minibatch[0][0]
        action_tens = torch.Tensor([minibatch[0][1]]).type(torch.LongTensor)
        reward_tens = torch.Tensor([[minibatch[0][2]]])
        observation_tens = minibatch[0][3]
        done_tens =torch.Tensor([[minibatch[0][4]]])
        adj_tens = self.graphs[minibatch[0][5]].adj().todense()
        adj_tens = torch.from_numpy(np.expand_dims(adj_tens.astype(int), axis=0)).type(torch.FloatTensor)


        for last_observation_, action_, reward_, observation_, done_, games_ in minibatch[-self.minibatch_length + 1:]:
            last_observation_tens=torch.cat((last_observation_tens,last_observation_))
            action_tens = torch.cat((action_tens, torch.Tensor([action_]).type(torch.LongTensor)))
            reward_tens = torch.cat((reward_tens, torch.Tensor([[reward_]])))
            observation_tens = torch.cat((observation_tens, observation_))
            done_tens = torch.cat((done_tens,torch.Tensor([[done_]])))
            adj_ = self.graphs[games_].adj().todense()
            adj = torch.from_numpy(np.expand_dims(adj_.astype(int), axis=0)).type(torch.FloatTensor)
            adj_tens = torch.cat((adj_tens, adj))

        return (last_observation_tens, action_tens, reward_tens, observation_tens,done_tens, adj_tens)

        
        
    def remember(self, last_observation, last_action, last_reward, observation,done):
        self.memory.append((last_observation, last_action, last_reward, observation,done, self.games))

    def remember_n(self,done):

        if not done:
            step_init = self.memory[-self.n_step]
            cum_reward=step_init[2]
            for step in range(1,self.n_step):
                cum_reward+=self.memory[-step][2]
            self.memory_n.append((step_init[0], step_init[1], cum_reward, self.memory[-1][-3],self.memory[-1][-2], self.memory[-1][-1]))

        else:
            for i in range(1,self.n_step+1):
                step_init = self.memory[-self.n_step+i]
                cum_reward=step_init[2]
                for step in range(1,self.n_step-i):
                    cum_reward+=self.memory[-step][2]
                if i==self.n_step-1:
                    self.memory_n.append(
                        (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], False, self.memory[-1][-1]))

                else:
                    self.memory_n.append((step_init[0], step_init[1], cum_reward,self.memory[-1][-3], False, self.memory[-1][-1]))

    
    
    
        
        
        
    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd+'/model_1200.pt')


Agent = DQAgent