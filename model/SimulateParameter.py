import networkx as nx
import networkit as nk
import numpy as np
from NetworkCreators import *
import pickle
import os

class SimulateParameter:
    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(self, graph, simulation_paras, start, end, state, county):
        # Store network graph (NetworkCreators object) and simulation params
        self.G = graph
        self.state = state
        self.county = county
        self.current_simulation_time = 0
        self.current_adoption_number = 0

        # p and q values (per income class group: low/mid/high)
        self.p_value = simulation_paras["p_value"]
        self.q_value = simulation_paras["q_value"]

        # Adoption history list over time
        self.adoption_history_list = []
        self.start = start
        self.end = end

        # Load empirical adoption data (state-level cumulative registrations)
        self.empirical_data = pd.read_csv(
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), '..', 'data', state, county, 'state_curve.csv')
            )
        )['cum_reg'].values

        # Load empirical adoption curves by group (low/mid/high)
        self.empirical_data_group = pickle.load(
            open(
                os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', state, county, 'emp_curve_group.pkl')),
                'rb'
            ),
            encoding='bytes'
        )
        
    # ------------------------------------------------------------
    # Run the simulation
    # ------------------------------------------------------------
    def run(self):
        # Simulate adoption over time steps
        for i in range(self.end):
            if i <= self.start:
                # Before or at start: record seed adopters only
                self.adoption_history_list.append(self.G.seed_num)
                self.current_adoption_number = self.G.seed_num
                self.current_simulation_time += 1
            else:
                # After start: perform state transition (new adoptions)
                self.transition()
                self.current_simulation_time += 1

    # ------------------------------------------------------------
    # One simulation step (adoption transition)
    # ------------------------------------------------------------
    def transition(self):
        adopted_node_list = []
        for node_id in range(self.G.current_node_number):
            if (self.G.node_attributes_attachment['adoption'][node_id] == 0):
                # Random threshold for adoption decision
                adopt_thr = np.random.random()
                # Class index: low=0, mid=1, high=2
                class_idx = int(self.G.node_attributes_attachment['class'][node_id])
                # Fraction of neighbors already adopted
                agent_num_neighbor_adopted = self.G.node_attributes_attachment['num_neighbor_adopted'][node_id]
                try:
                    network_value = agent_num_neighbor_adopted / self.G.node_attributes_attachment['degree'][node_id]
                except:
                    network_value = 0
                # Adoption probability = p + q * (fraction neighbors adopted)
                p = self.p_value[class_idx] + self.q_value[class_idx] * network_value
                if p > adopt_thr:
                    adopted_node_list.append(node_id)

        # Update adoption state for new adopters
        for node_id in adopted_node_list:
            self.G.node_attributes_attachment['adoption'][node_id] = 1
            self.G.node_attributes_attachment["adoption_time"][node_id] = self.current_simulation_time
            self.current_adoption_number += 1
            # Increment neighbor adoption counts
            for neighbor in self.G.iterNeighbors(node_id):
                self.G.node_attributes_attachment['num_neighbor_adopted'][neighbor] += 1

        # Append adoption count to history
        self.adoption_history_list.append(self.current_adoption_number)
    
    # ------------------------------------------------------------
    # Error calculation (state-level)
    # ------------------------------------------------------------
    def calculate_absolute_error(self):
        model_data = np.array(self.adoption_history_list) / self.G.scale
        error = np.sum(abs(self.empirical_data[self.start:self.end] - model_data[self.start:self.end])**2)
        return error / self.end
    
    # ------------------------------------------------------------
    # Error calculation (group-level)
    # ------------------------------------------------------------
    def calculate_absolute_error_group(self):
        group_num = 3
        adoption_sim_group = {}
        adoption_emp_group = self.empirical_data_group
        error_all = []
        for group_id in range(group_num):
            # Initialize simulated group curve
            adoption_sim_group[group_id] = [0] * self.end
            for node_id in range(self.G.current_node_number):
                if (self.G.node_attributes_attachment['class'][node_id] == group_id) and \
                   (self.G.node_attributes_attachment["adoption_time"][node_id] > -1):
                    adoption_sim_group[group_id][self.G.node_attributes_attachment["adoption_time"][node_id]] += 1
            # Cumulative simulated adoption
            model_data = np.cumsum(np.array(adoption_sim_group[group_id]) / self.G.scale)[self.start:]
            empirical_data = adoption_emp_group[group_id][self.start:self.end]
            # Squared error for this group
            error_all.append(np.sum((model_data - empirical_data)**2))
        return np.max(error_all) / self.end

    # ------------------------------------------------------------
    # Reset simulation
    # ------------------------------------------------------------
    def reset(self):
        self.adoption_history_list = []
        self.current_simulation_time = 0
        self.current_adoption_number = 0
        self.G.reset()

    # ------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------
    def output_curve_by_state(self):
        # Return simulated adoption curve (state level)
        return self.adoption_history_list

    def output_curve_by_tract(self):
        # Return simulated adoption curve aggregated by tract
        zipcode_dict = {}
        for node_id in range(self.G.current_node_number):
            if self.G.node_attributes_attachment['tract'][node_id] not in zipcode_dict:
                zipcode_dict[self.G.node_attributes_attachment['tract'][node_id]] = [0] * self.end
            if self.G.node_attributes_attachment['adoption'][node_id] == 1:
                zipcode_dict[self.G.node_attributes_attachment['tract'][node_id]][
                    self.G.node_attributes_attachment["adoption_time"][node_id]] += 1
        return pd.DataFrame.from_dict(zipcode_dict)
    
    def output_curve_by_agent(self):
        # Return adoption time for each individual node
        zipcode_dict = {}
        for node_id in range(self.G.current_node_number):
            zipcode_dict[node_id] = self.G.node_attributes_attachment["adoption_time"][node_id]
        return pd.DataFrame.from_dict(zipcode_dict, orient='index', columns=['adoption_time'])

    def output_error_by_state(self):
        # Return group-based error metric
        return self.calculate_absolute_error_group()



