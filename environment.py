import numpy as np
import torch
import graph
import bus_line

import time

"""
This file contains the definition of the environment
in which the agents are run.
"""

class Environment:
    
    def __init__(self, graph):
        self.graphs = graph

    def reset(self, g, epoch_, graph_num):        
        self.epoch_ = epoch_
        self.games = graph_num
        self.graph_num = graph_num
        self.graph_init = self.graphs[self.graph_num]
        self.nodes = self.graph_init.nodes()
        self.nbr_of_nodes = 0
        self.edge_add_old = 0
        self.last_reward = 0
        
        
        # Set observation
        self.observation  = torch.ones(1, self.nodes, 1, dtype=torch.float)


        self.initial_result = self.get_reward(self.observation)
        self.initial_acc = self.initial_result[0]
        self.initial_frequency = self.initial_result[2]
        self.initial_list_acc = self.initial_result[3]
        self.initial_Akinson = self.initial_result[4]
        self.initial_Pietra = self.initial_result[5]
        self.initial_Theil = self.initial_result[6]
        
        
        self.previous_acc = self.initial_acc
        
        #set patient
        self.patient = 25
        self.patient_acc_max = self.initial_acc
        

        if g == 1:
            self.mode = 'train'
        elif g == 2:
            self.mode = 'RLtest'
        else:
            self.mode = 'RDtest'
        
    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation

    def act(self,node):
        #calculate old acc
        acc_old = self.previous_acc       
        
        #update observation
        self.observation[:,node,:]=0

        
        acc_new = self.get_reward(self.observation)
        self.previous_acc = acc_new[0]
        #cheak the termination and get reward
        if acc_new[0] > self.patient_acc_max:
            self.patient_acc_max = acc_new[0]
            self.patient = 25
        else:
            self.patient = self.patient-1

        if self.patient == 0:
            reward = (acc_new[0] -acc_old,True,acc_new[2],acc_new[3],acc_new[4],acc_new[5],acc_new[6])
        else:
            reward = (acc_new[0] -acc_old,acc_new[1],acc_new[2],acc_new[3],acc_new[4],acc_new[5],acc_new[6])
        
#        print('      chosen node :',node)
#        print('      reward :',reward[0])
#        print('      finish?',reward[1])
      
        
        return reward


    def get_reward(self, observation):
        
        obser =  np.reshape(observation.numpy(),-1) 
        
        node_index = np.array([i for i in range(self.nodes)])[ obser != 0 ]
        
#         print(node_index)
        result = self.f( node_index , self.graph_init)
        reward = result[0]
        g_frequency = result[1]
        g_list_acc = result[2]
        g_Akinson = result[3]
        g_Pietra = result[4]
        g_Theil = result[5]
        
        done = False
        
#         if len(node_index)==50:
#             done = True
        if len(node_index)==2:
            done = True
        return [reward,done,g_frequency,g_list_acc,g_Akinson,g_Pietra,g_Theil]
    
    def f( self , array_a , g_initial ):
        map_ = g_initial.all_pos
        map_1 = {}
        map_2 = {}
        map_3 = {}
        map_4 = {}
        map_5 = {}
        map_6 = {}
        map_7 = {}
        map_8 = {}
        line_1 = array_a[ array_a <10 ]
        for i in line_1:
            map_1[i] = map_[i]

        array_b = array_a[ array_a >=10 ]
        line_2 = array_b[ array_b <20 ]
        for i in line_2:
            map_2[i] = map_[i]

        array_b = array_a[ array_a >=20 ]
        line_3 = array_b[ array_b <30 ]
        for i in line_3:
            map_3[i] = map_[i]

        array_b = array_a[ array_a >=30 ]
        line_4 = array_b[ array_b <40 ]
        for i in line_4:
            map_4[i] = map_[i]
            
        array_b = array_a[ array_a >=40 ]
        line_5 = array_b[ array_b <50 ]
        for i in line_5:
            map_5[i] = map_[i]
            
        array_b = array_a[ array_a >=50 ]
        line_6 = array_b[ array_b <60 ]
        for i in line_6:
            map_6[i] = map_[i]
            
        array_b = array_a[ array_a >=60 ]
        line_7 = array_b[ array_b <70 ]
        for i in line_7:
            map_7[i] = map_[i] 
            
        array_b = array_a[ array_a >=70 ]
        line_8 = array_b[ array_b <80 ]
        for i in line_8:
            map_8[i] = map_[i]
        #create  bus_line
        bus_line_1 = bus_line.Bus_line( 1,1, list(line_1), map_1)
        bus_line_2 = bus_line.Bus_line( 2,1, list(line_2), map_2)
        bus_line_3 = bus_line.Bus_line( 3,1, list(line_3), map_3)
        bus_line_4 = bus_line.Bus_line( 4,1, list(line_4), map_4)
        bus_line_5 = bus_line.Bus_line( 5,1, list(line_5), map_5)
        bus_line_6 = bus_line.Bus_line( 6,1, list(line_6), map_6)
        bus_line_7 = bus_line.Bus_line( 7,1, list(line_7), map_7)
        bus_line_8 = bus_line.Bus_line( 8,1, list(line_8), map_8)
        #create graph
        g  = graph.Graph([1],[1])
        #add each bus_line
        g.add_bus_line(bus_line_1)
        g.add_bus_line(bus_line_2)
        g.add_bus_line(bus_line_3)
        g.add_bus_line(bus_line_4)
        g.add_bus_line(bus_line_5)
        g.add_bus_line(bus_line_6)
        g.add_bus_line(bus_line_7)
        g.add_bus_line(bus_line_8)
        for connection in g_initial.list_connection:
            if connection[0] in array_a and connection[1] in array_a:
                travel_time = np.linalg.norm(np.array( g_initial.all_pos[connection[0]] )-np.array( g_initial.all_pos[connection[1]] ))*100/300
                g.add_connection( [ ( connection[0],connection[1],travel_time+g_initial.list_waiting_time[connection[1]//10 ])] ) 
                g.add_connection([(connection[1],connection[0],travel_time+g_initial.list_waiting_time[connection[0]//10])]) 

        g.add_center()
        g.add_edge_between_centers()
        g.add_edge_between_centerAnsBusStation()
        acc = g.get_acc()
        g_frequency = g.avg_frequency()
        g_acc = acc[0]
        g_list_acc = acc[1]
        g_Akinson = acc[2]
        g_Pietra = acc[3]
        g_Theil = acc[4]
        return [g_acc,g_frequency,g_list_acc,g_Akinson,g_Pietra,g_Theil]

    