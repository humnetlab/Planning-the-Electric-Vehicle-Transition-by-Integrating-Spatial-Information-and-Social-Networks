import networkx as nx
import networkit as nk
import numpy as np
from NetworkCreators import *
import pickle
import os

class SimulateParameter:
    def __init__(self, graph, simulation_paras, start, end, state, county):
        self.G = graph
        self.state = state
        self.county = county
        self.current_simulation_time = 0
        self.current_adoption_number = 0
        self.p_value = simulation_paras["p_value"]
        self.q_value = simulation_paras["q_value"]
        self.return_year = simulation_paras["return_year"]
        self.return_prob = simulation_paras["return_prob"]
        self.adoption_history_list = []
        self.start = start
        self.end = end
        self.empirical_data = pd.read_csv(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', state, county, 'state_curve.csv')))['cum_reg'].values
        self.empirical_data_group = pickle.load(open(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', state, county, 'emp_curve_group.pkl')), 'rb'), encoding='bytes')
        
    def run(self):
        for i in range(self.end):
            if i <= self.start:
                self.adoption_history_list.append(self.G.seed_num)
                self.current_adoption_number = self.G.seed_num
                self.current_simulation_time += 1
            else:
                self.transition()
                self.current_simulation_time += 1


    def transition(self):
        adopted_node_list = []; noadopted_node_list = []
        for node_id in range(self.G.current_node_number):
            # if (self.G.node_attributes_attachment['adoption'][node_id] == 0) and (self.G.node_attributes_attachment['degree'][node_id]>0):
            if (self.G.node_attributes_attachment['adoption'][node_id] == 1) and (self.current_simulation_time-self.G.node_attributes_attachment["adoption_time"][node_id]>self.return_year):
                return_thr = np.random.random()
                if self.return_prob > return_thr:
                    noadopted_node_list.append(node_id)
                
            if (self.G.node_attributes_attachment['adoption'][node_id] == 0):
                adopt_thr = np.random.random()
                class_idx = int(self.G.node_attributes_attachment['class'][node_id])
                agent_num_neighbor_adopted = self.G.node_attributes_attachment['num_neighbor_adopted'][node_id]
                try:
                    network_value = agent_num_neighbor_adopted/self.G.node_attributes_attachment['degree'][node_id]
                except:
                    network_value = 0
                p = self.p_value[class_idx] + self.q_value[class_idx] * network_value
                if p > adopt_thr:
                    adopted_node_list.append(node_id)
            
        for node_id in noadopted_node_list:
            self.G.node_attributes_attachment['adoption'][node_id] = 0
            self.G.node_attributes_attachment["adoption_time"][node_id] = -1
            self.current_adoption_number -= 1
            for neighbor in self.G.iterNeighbors(node_id):
                self.G.node_attributes_attachment['num_neighbor_adopted'][neighbor] -= 1

        for node_id in adopted_node_list:
            self.G.node_attributes_attachment['adoption'][node_id] = 1
            self.G.node_attributes_attachment["adoption_time"][node_id] = self.current_simulation_time
            self.current_adoption_number += 1
            for neighbor in self.G.iterNeighbors(node_id):
                self.G.node_attributes_attachment['num_neighbor_adopted'][neighbor] += 1
                
        self.adoption_history_list.append(self.current_adoption_number)
        # print(self.current_adoption_number, len(noadopted_node_list), len(adopted_node_list))
    
    def calculate_absolute_error(self):
        model_data = np.array(self.adoption_history_list) / self.G.scale
        error = np.sum(abs(self.empirical_data[self.start:self.end] - model_data[self.start:self.end])**2)
        return error/self.end
    
    def calculate_absolute_error_group(self):
        group_num = 3
        adoption_sim_group = {}
        adoption_emp_group = self.empirical_data_group
        error_all = []
        for group_id in range(group_num):
            adoption_sim_group[group_id] = [0] * self.end
            for node_id in range(self.G.current_node_number):
                if (self.G.node_attributes_attachment['class'][node_id] == group_id) and (self.G.node_attributes_attachment["adoption_time"][node_id]>-1):
                    adoption_sim_group[group_id][self.G.node_attributes_attachment["adoption_time"][node_id]] += 1
            model_data = np.cumsum(np.array(adoption_sim_group[group_id])/self.G.scale)[self.start:]
            empirical_data = adoption_emp_group[group_id][self.start:self.end]
            error_all.append(np.sum((model_data-empirical_data)**2))
        return np.max(error_all)/self.end

    def reset(self):
        self.adoption_history_list = []
        self.current_simulation_time = 0
        self.current_adoption_number = 0
        self.G.reset()

    def output_curve_by_state(self):# check
        return self.adoption_history_list

    def output_curve_by_tract(self):# check
        zipcode_dict = {}
        for node_id in range(self.G.current_node_number):
            if self.G.node_attributes_attachment['tract'][node_id] not in zipcode_dict:
                zipcode_dict[self.G.node_attributes_attachment['tract'][node_id]] = [0] * self.end
            if self.G.node_attributes_attachment['adoption'][node_id] == 1:
                zipcode_dict[self.G.node_attributes_attachment['tract'][node_id]][
                    self.G.node_attributes_attachment["adoption_time"][node_id]] += 1
        return pd.DataFrame.from_dict(zipcode_dict)
    
    def output_curve_by_agent(self):# check
        zipcode_dict = {}
        for node_id in range(self.G.current_node_number):
            zipcode_dict[node_id] = self.G.node_attributes_attachment["adoption_time"][node_id]
        return pd.DataFrame.from_dict(zipcode_dict,orient='index', columns=['adoption_time'])

    def output_error_by_state(self):# check
        return self.calculate_absolute_error_group()

    def update_num_neighbor_adopted(self):
        for node in self.iterNodes():
            num_neighbor_adopted = 0
            for neighbor in self.iterNeighbors(node):
                if self.node_attributes_attachment['adoption'][neighbor] == 1 and node != neighbor:
                    num_neighbor_adopted += 1
            self.node_attributes_attachment['num_neighbor_adopted'][node] = num_neighbor_adopted

