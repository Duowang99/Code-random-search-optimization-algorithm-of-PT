"""
This is the machinnery that runs your agent in an environment.
"""
import matplotlib.pyplot as plt
import numpy as np
import agent
import matplotlib.pyplot as plt
import random

class Runner:
    def __init__(self, environment, agent, phrase ):
        self.environment = environment
        self.agent = agent
        self.phrase = phrase
    def step(self):
        observation = self.environment.observe().clone()
        #agent makes an action
        action = self.agent.act(observation)
        old_feature = 1#self.environment.feature[action]
        #get reward from environment
        (reward, done, frequency,g_list_acc,g_Akinson,g_Pietra,g_Theil) = self.environment.act(action)
        observation_new = self.environment.observe().clone()
        return (observation, action, reward, done, frequency,old_feature,g_list_acc,g_Akinson,g_Pietra,g_Theil,observation_new)

    def loop(self, nbr_epoch, max_iter):
        train_list_cumul_reward=[]
        test_list_min_step=[]
        train_list_min_acc = []
        test_list_min_df_acc = []
        test_list_frequency = []
        test_list_feature_corr = []
        test_list_acc = []
        test_list_Akinson = []
        test_list_Pietra = []
        test_list_Theil = []
        test_list_obs = []
        
        Q_change_rate_list = []
        
        train_relation_patient = []
        for epoch_ in range(nbr_epoch):
            print(" -> epoch : "+str(epoch_))
            if self.phrase == 'train':
                g = 1
                graph_num = epoch_
            elif self.phrase == 'RLtest':
                g = 2
                graph_num = epoch_ 
            else:#'RDtest'
                g = 3
                graph_num = epoch_ 
                
            if g == 1:
                print(' --> train phrase ',g)
            else:
#                print(' --> test phrase ')
                test_feature = []
                test_reward = []
                test_min_acc = []
                list_frequency = []
                list_acc = []
                list_Akinson = []
                list_Pietra = []
                list_Theil = []
                test_cumul_reward = 0.0
                self.environment.reset(g, epoch_, graph_num)
                self.agent.reset(g, epoch_, graph_num)
                list_obs = []
                

                test_min_acc.append( self.environment.initial_acc )
                list_frequency.append( self.environment.initial_frequency )
                list_acc.append( self.environment.initial_list_acc   )
                list_Akinson.append( self.environment.initial_Akinson )
                list_Pietra.append( self.environment.initial_Pietra )
                list_Theil.append( self.environment.initial_Theil )
                list_obs.append( self.environment.observe().clone() )
                
                for i in range(1, max_iter + 1):
#                    print(" --> step : "+str(i))
                    (obs, act, rew, done, frequency,old_feature,g_list_acc,g_Akinson,g_Pietra,g_Theil,observation_new) = self.step()
                    test_cumul_reward += rew
                    test_min_acc.append( test_cumul_reward+self.environment.initial_acc )
                    test_feature.append( old_feature )
                    test_reward.append( rew )
                    list_frequency.append( frequency )
                    list_acc.append( g_list_acc )
                    list_Akinson.append( g_Akinson )
                    list_Pietra.append( g_Pietra )
                    list_Theil.append( g_Theil )
                    list_obs.append( observation_new )
                    if done: 
#                        print('      min accessibility :',  np.max(test_min_acc)  )
#                         print( np.mean(np.array(list_acc)[:,:10],axis=1 ))
#                         print(len(test_min_acc))
#                         print(len(list_Akinson))
#                         print(len(list_Pietra))
#                         print(len(list_Theil))
#                         print(len( list_obs ))
                        test_list_min_step.append( np.argmax( test_min_acc ) )        
                        test_list_min_df_acc.append(( np.max( test_min_acc )-self.environment.initial_acc )/self.environment.initial_acc)
                        test_list_frequency.append( list_frequency[ np.argmax( test_min_acc ) ] )
                        test_list_acc.append( list_acc[np.argmax( test_min_acc )] )
                        test_list_Akinson.append( list_Akinson[np.argmax( test_min_acc )] )
                        test_list_Pietra.append( list_Pietra[np.argmax( test_min_acc )] )
                        test_list_Theil.append( list_Theil[np.argmax( test_min_acc )] )
                        test_list_obs.append( list_obs[np.argmax( test_min_acc )] )

                        
                        
                        break
                
        
        return [test_list_min_step,test_list_min_df_acc,test_list_frequency,test_list_acc,test_list_Akinson,test_list_Pietra,test_list_Theil,test_list_obs]
    
    
    
    
    
    
    
    
    
def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))
