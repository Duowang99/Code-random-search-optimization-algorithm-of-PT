import numpy as np
import networkx as nx
import collections
import bus_line
import matplotlib.pyplot as plt

# seed = np.random.seed(120)

def edgeOfBetweenCenters(a):
    node_1,node_2,node_3,node_4 = a[0],a[1],a[2],a[3]
    edge_list =[(node_1,node_2,5.0*100/60),(node_2,node_1,5.0*100/60)
                ,(node_2,node_3,5.0*100/60),(node_3,node_2,5.0*100/60)
                ,(node_3,node_4,5.0*100/60),(node_4,node_3,5.0*100/60)
                ,(node_4,node_1,5.0*100/60),(node_1,node_4,5.0*100/60)
               ,(node_1,node_3,7.0*100/60),(node_3,node_1,7.0*100/60)
               ,(node_2,node_4,7.0*100/60),(node_4,node_2,7.0*100/60)]
    return edge_list

def edgeCenterAndBusStation(center,bus_pos,all_pos,waiting_time):
    bus_stop_name = list( bus_pos.keys())
    bus_pos_list = list( bus_pos.values() )
    center_pos = np.array( all_pos[center] )
    list_out = []
    for i in range(len(bus_stop_name)):
        stop_name = bus_stop_name[i]
        dis_center_bus = np.linalg.norm(center_pos-np.array(bus_pos_list[i]))*100/60
        list_out.append( (center,stop_name,dis_center_bus+waiting_time[stop_name]) )
        list_out.append( (stop_name,center,dis_center_bus) )
    return list_out

class Graph:
    
    def __init__(self, list_connection,list_waiting_time):
        self.g = nx.DiGraph()
        self.bus_node = []
        self.bus_pos = {}
        self.center_node = []
        self.center_pos = []
        self.all_node = []
        self.all_pos = {}
        self.number_of_BusStations = 0
        self.bus_waiting_time = {}
        
        self.list_connection = list_connection
        self.list_waiting_time = list_waiting_time
        
        self.list_frequency = []
        
    def add_bus_line(self,bus_line):
        self.g.add_nodes_from(bus_line.bus_stop_list)
        self.g.add_weighted_edges_from(bus_line.line)#
        self.bus_node += bus_line.bus_stop_list
        self.bus_pos = {**self.bus_pos.copy(), **bus_line.bus_stop_pos}.copy()
        self.all_node = self.bus_node.copy()
        self.all_pos = self.bus_pos.copy()
        self.number_of_BusStations += len( bus_line.bus_stop_list)
        self.bus_waiting_time = {**self.bus_waiting_time.copy(), **dict.fromkeys(bus_line.bus_stop_list,bus_line.waiting_time)}.copy()
        
        self.list_frequency.append( bus_line.frequency )
        
    def add_connection(self, list_connection ):
        self.g.add_weighted_edges_from(list_connection)
        
    def add_center(self):
        center_node = [ i+80 for i in range(0,100) ]
        center_pos = np.reshape([[ (i*5,j*5) for j in range(10) ] for i in range(10)],(100,2))
        self.g.add_nodes_from(center_node)
        self.center_node = center_node
        self.center_pos = center_pos
        self.all_node +=  center_node
        for i in range(len(self.center_node)):
            self.all_pos[  center_node[i] ] = tuple(center_pos[i]) 
            
    def add_edge_between_centers(self):
        ss = [i for i in range(80, 169)]
        ss.remove(89)
        ss.remove(99)
        ss.remove(109)
        ss.remove(119)
        ss.remove(129)
        ss.remove(139)
        ss.remove(149)
        ss.remove(159)
        point_list = [[i,i+10,i+11,i+1] for i in ss]
        list_edge_a = []
        for point in point_list:
            list_edge_a+= edgeOfBetweenCenters(point)
        list_edge_a = list(set(list_edge_a))
        self.g.add_weighted_edges_from(list_edge_a)
    
    def add_edge_between_centerAnsBusStation(self):
        list_edge = []
        for i in self.center_node:
            list_edge +=  edgeCenterAndBusStation(i,self.bus_pos,self.all_pos,self.bus_waiting_time)
        self.g.add_weighted_edges_from(list_edge)
    
    def show(self):
        fig=plt.figure(figsize=(10,10))
        node_color=["r" for i in range(self.number_of_BusStations)] + ['b' for i in range(len(self.center_node)) ]
        node_size=[50 for i in range(self.number_of_BusStations)] + [10 for i in range(len(self.center_node)) ]
        nx.draw(self.g, self.all_pos, with_labels=True, node_color=node_color, node_size = node_size)
        plt.show()
        
    def get_acc(self):
        #centroid i from 40 to 139, population of i
        center_pos_0 = np.reshape([[ (i*5,j*5) for j in range(10) ] for i in range(10)],(100,2))
        center_polulation = {}
        for i in range(len(center_pos_0)):
            r = np.linalg.norm(center_pos_0[i]-np.array([22.5,22.5]))
            center_polulation[i+80] = 36000*np.exp(-0.01*r)/4
        total_polulation = np.sum( list(center_polulation.values()) )
        #claculate inacc
        length =  nx.all_pairs_dijkstra_path_length(self.g)
        list_acc = []
        for i, dict_ in length:
            if i in self.center_node:
                acc_i = 0.0
                for j in dict_.keys():
                    if j in self.center_node and j != i:
                        acc_i = acc_i + center_polulation[j]/dict_[j]
                list_acc.append(acc_i)
        list_acc_0 = list_acc.copy()
        list_acc_0.sort()
        
        #Akinson
        y_mean = np.mean( list_acc )   
        sum_ = 0.
        for i in range( len(list_acc) ):
            sum_ = sum_ + list_acc[i]**(-1) 
        sum_ = 1 - (sum_/len(list_acc))**(-1)/y_mean
        
        #Pietra     
        sum_P = 0.
        for i in range( len(list_acc) ):
            sum_P = sum_P +  np.abs(list_acc[i]-y_mean)/y_mean
        sum_P = sum_P/(2* len(list_acc) )
        
        #Theil
        sum_T = 0.
        for i in range( len(list_acc) ):
            sum_T = sum_T +  list_acc[i]/y_mean*np.log( list_acc[i]/y_mean )
        sum_T = sum_T/len(list_acc)
        
        return [np.mean(list_acc_0),list_acc,sum_,sum_P,sum_T]
#         sum_ = 0.
#         for i in range( len(list_acc) ):
#             sum_ = sum_ + list_acc[i]**(0.5)      
#         return [sum_**2,list_acc]
    
    
    
    
    def avg_frequency(self):
        return self.list_frequency
    
    
    
    def nodes(self):
        
        return nx.number_of_nodes(self.g)

    def edges(self):

        return self.g.edges()

    def neighbors(self, node):

        return nx.all_neighbors(self.g,node)

    def average_neighbor_degree(self, node):

        return nx.average_neighbor_degree(self.g, nodes=node)

    def adj(self):

        return nx.adjacency_matrix(self.g)
