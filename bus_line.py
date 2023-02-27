import networkx as nx
import pylab 
import numpy as np
import matplotlib.pyplot as plt





class Bus_line:
    '''Parameters:  number : the line number of this bus line. e.g. 9106
                            number_of_bus : bus number of this bus line
                            bus_stop_list : all the bus stop ids by a list
                            bus_stop_pos : all the bus stop positions by dict{ bus_stop_id: (x,y) }
    '''
    def __init__(self, number,number_of_bus, bus_stop_list, bus_stop_pos):
        self.number = number
        self.bus_stop_list = bus_stop_list
        self.bus_stop_pos = bus_stop_pos
        self.number_of_bus = number_of_bus
        self.line_length = len( bus_stop_list ) -1
        
        self.times = []
        self.line = []
        
        #计算等车时间
        for i in range(self.line_length):
            start_id = self.bus_stop_list[i]
            stop_id = self.bus_stop_list[i+1]
            time = np.linalg.norm(np.array(bus_stop_pos[start_id])-np.array(bus_stop_pos[stop_id]))*100/300
            self.times.append(time)       
        # end-to-end time np.sum(times)
        self.waiting_time = np.sum(self.times)/self.number_of_bus
        
        self.frequency = ( np.sum(self.times)+self.line_length )/self.number_of_bus
        
        #加上从start_id到stop_id所用时间
        for i in range(self.line_length):
            start_id = self.bus_stop_list[i]
            stop_id = self.bus_stop_list[i+1]
            time = np.linalg.norm(np.array(start_id)-np.array(stop_id))*100/300 + 1 #Dwell time=1min
            self.line.append( ( start_id, stop_id, time ) )
            self.line.append( ( stop_id, start_id, time ) )
            #times.append(time)