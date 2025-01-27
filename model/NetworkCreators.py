import numpy as np
import pandas as pd
import networkit as nk
import networkx as nx
import copy
import os
import pickle
import matplotlib.pyplot as plt
import powerlaw

# Three group based on income distribution
def generate_class(input_args):
    classid = input_args['class']
    state = input_args['state']
    if classid == 'High':
        return 2
    elif classid == 'Middle':
        return 1
    else:
        return 0    

class NetworkCreators(nk.graph.Graph):
    def __init__(self, n, par):
        super().__init__()
        self.state = par[8]
        self.county = par[9]
        self.current_node_number = 0
        self.homo = par[0]
        self.r_exp = par[1]
        self.k_exp = par[2]
        self.k_min = par[3]
        self.start = par[4]
        self.designed_node_number = par[7]
        self.par = par
        self.node_attributes_attachment = {}
        self.node_attribute_dict = {}
        self.node_list = []
        self.node_tract = []
        self.seed_node_reset = {}
        self.TRACT_COORDINATES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',self.state, self.county,'tract_coord.csv')
        self.M_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', self.state, self.county, 'M.npy')
        self.DEMO_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', self.state, self.county, 'demo_data.csv'))
        self.RESULT_PATH = os.path.join(os.path.dirname(__file__), '..', 'result',self.state, self.county)
        self.tract_idx_dict = pd.read_csv(self.TRACT_COORDINATES_PATH).reset_index().set_index('GEOID')['index'].to_dict()
        if not os.path.exists(self.RESULT_PATH):
            os.makedirs(self.RESULT_PATH)

    def set_scale_value(self, true_value, model_value):
        self.scale = model_value/true_value
    
    def generate_node_initial(self):
        attribute_dict = {
            "class": generate_class,
            "adoption": 0,
            "tract": ' ',
            "county": ' ',
            "degree": 0,
            "num_neighbor_adopted": 0,
            "adoption_time": -1
        }
        
        attribute_type_dict ={
            "class": int,
            "adoption": int,
            "tract": str,
            "county": str,
            "degree": int,
            "num_neighbor_adopted": int,
            "adoption_time": int
        }
        self.node_attribute_dict = attribute_dict
        for key in attribute_dict:
            self.node_attributes_attachment[key] = self.attachNodeAttribute(key, attribute_type_dict[key])

    def generate_nodes_attribute(self, number: int, attribute_dict, **kwargs):
        for node_id in range(self.current_node_number, self.current_node_number + number):
            self.addNode()
            for key, function in attribute_dict.items():
                if callable(function):
                    value = function(**kwargs)
                else:
                    value = function
                self.node_attributes_attachment[key][node_id] = value
            self.node_list += [int(self.node_attributes_attachment['class'][node_id])]
            self.node_tract += [self.tract_idx_dict[int(self.node_attributes_attachment['tract'][node_id])]]
        self.current_node_number += number

    def generate_nodes(self):
        self.generate_node_initial()
        data = pd.read_csv(self.DEMO_PATH,converters={'GEOID': str})
        population_sum = sum(data['POPULATION'].astype(int))
        self.set_scale_value(population_sum, self.designed_node_number)

        for i in range(len(data)):
            item = data.iloc[i]
            number = int(np.round(int(item['POPULATION']) * self.scale))
            temp_attribute_dict = copy.copy(self.node_attribute_dict)
            temp_attribute_dict.update({"tract": str(item['GEOID'])})
            temp_attribute_dict.update({"county": str(item['COUNTY'])})
            input_args = {"class": item['CLASS'], "state":self.state}
            self.generate_nodes_attribute(number, temp_attribute_dict, input_args = input_args)

    def generate_edge_list(self):
        if os.path.isfile(
            os.path.join(self.RESULT_PATH,'edge_list_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy')
        ):  
            print('edge list found...')
            self.edge_list = np.load(os.path.join(self.RESULT_PATH,'edge_list_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy'))
            D = np.load(self.M_PATH)
            self.D = D
            self.node_tract = np.array(self.node_tract)
            return

        print('generating edge list...')
        edge_list = []
        D = np.load(self.M_PATH)
        self.D = D
        self.node_tract = np.array(self.node_tract)
        
        n_node = self.current_node_number; n_rep = 3; rc = 1
        nodes = list(range(n_node))

        active_nodes = np.random.choice(nodes,n_rep) 
        inactive_nodes = list(set(nodes)-set(active_nodes))
        K = np.zeros(n_node)
        R = []
        edges = []

        while(inactive_nodes):
            inactive_node = np.random.choice(inactive_nodes,1)[0]
            inactive_node_tract = self.node_tract[inactive_node]
            active_nodes_tract = self.node_tract[active_nodes]
            if self.r_exp < 0:
                p_vec_norm = (K[active_nodes]+1)/np.exp(D[inactive_node_tract,active_nodes_tract]/abs(self.r_exp))
            else:
                p_vec_norm = (K[active_nodes]+1)/(D[inactive_node_tract,active_nodes_tract])**self.r_exp
            p_vec_norm = p_vec_norm/sum(p_vec_norm)

            active_node_p = np.random.choice(active_nodes, n_rep, p = p_vec_norm, replace=False)
            for active_node in active_node_p:
                edges.append((inactive_node,active_node))
                R.append(D[self.node_tract[inactive_node],self.node_tract[active_node]])
                K[inactive_node] = K[inactive_node]+1; K[active_node] = K[active_node]+1
                inactive_nodes = list(set(inactive_nodes)-set([inactive_node]))
                active_nodes = list(set(active_nodes)|set([inactive_node]))
        
        self.edge_list = edges
        
        G=nx.Graph()
        G.add_edges_from(self.edge_list)
        for node in self.iterNodes():
            if self.node_attributes_attachment['class'][node] == 0:
                G.add_node(node,classes='low')
            if self.node_attributes_attachment['class'][node] == 1:
                G.add_node(node,classes='mid')
            if self.node_attributes_attachment['class'][node] == 2:
                G.add_node(node,classes='high')
        H = nx.attribute_assortativity_coefficient(G,'classes')   
        C = pd.DataFrame.from_dict(nx.clustering(G), orient='index').sort_index()[0].values
        M = nx.attribute_mixing_matrix(G, 'classes', mapping={'low': 0, 'mid': 1, 'high':2})
        print('H+C')
        print(H,nx.average_clustering(G))
        np.save(os.path.join(self.RESULT_PATH, 'M_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy'), np.array(M))
        np.save(os.path.join(self.RESULT_PATH, 'K_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy'), np.array(K))
        np.save(os.path.join(self.RESULT_PATH, 'R_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy'), np.array(R))
        np.save(os.path.join(self.RESULT_PATH, 'C_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy'), np.array(C))
        np.save(os.path.join(self.RESULT_PATH, 'H_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy'), np.array(H))
        np.save(os.path.join(self.RESULT_PATH, 'edge_list_'+str(self.designed_node_number)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)+'.npy'), np.array(self.edge_list))
        del G
        
    def generate_edges(self):
        self.generate_edge_list()
        for row in self.edge_list:
            source = row[0]
            target = row[1]
            self.addEdge(source, target)
        # print(f'number of edges = {len(self.edge_list)}')
    
    def set_node_degree(self):
        for node in self.iterNodes():
            self.node_attributes_attachment['degree'][node] = self.degree(node)
    
    def update_num_neighbor_adopted(self):
        for node in self.iterNodes():
            num_neighbor_adopted = 0
            for neighbor in self.iterNeighbors(node):
                if self.node_attributes_attachment['adoption'][neighbor] == 1 and node != neighbor:
                    num_neighbor_adopted += 1
            self.node_attributes_attachment['num_neighbor_adopted'][node] = num_neighbor_adopted

    def create_seed(self):
        if self.start<0:
            start = 12
        else:
            start = self.start
        self.seed_num = 0; self.seed_type = 'tract'
        seed_emp = (pd.read_csv(os.path.join( '..', 'data', self.state, self.county, self.seed_type+'_curve.csv'),converters={'tract': str}))
        seed_emp = seed_emp[seed_emp['year'] == start]
        seed_emp['cum_reg'] = (seed_emp['cum_reg']*self.scale).round(0)
        self.seed_node = dict(zip(seed_emp[self.seed_type].astype(str), seed_emp['cum_reg'])).copy()
        self.seed_node_reset = dict(zip(seed_emp[self.seed_type].astype(str), seed_emp['cum_reg'])).copy()

    def update_num_neighbor_adopted(self):
        for node in self.iterNodes():
            num_neighbor_adopted = 0
            for neighbor in self.iterNeighbors(node):
                if self.node_attributes_attachment['adoption'][neighbor] == 1 and node != neighbor:
                    num_neighbor_adopted += 1
            self.node_attributes_attachment['num_neighbor_adopted'][node] = num_neighbor_adopted

    def cal_class_degree(self):
        node_list = []; low_list = []; middle_list=[]; high_list=[]
        for node in self.iterNodes():
            node_list.append(node)
            low = 0; middle = 0; high = 0
            for neighbor in self.iterNeighbors(node):
                if self.node_attributes_attachment['class'][neighbor] == 0:
                    low += 1
                if self.node_attributes_attachment['class'][neighbor] == 1:
                    middle += 1
                if self.node_attributes_attachment['class'][neighbor] == 2:
                    high += 1      
            low_list.append(low)
            middle_list.append(middle)
            high_list.append(high)
        df = pd.DataFrame({'id':node_list,'low':low_list,'middle':middle_list,'high':high_list})
        df.to_csv('degree_'+str(self.r_exp)+'.csv',index=False)
        self.class_focus = self.par[6]
        return df

    def reset(self):
        self.seed_node = self.seed_node_reset.copy()
        seed_adopters_num = 0

        for key1 in self.seed_node_reset.keys():
            seed_adopters_num += self.seed_node_reset[key1]
        self.seed_num = seed_adopters_num

        for node in range(self.current_node_number):
            self.node_attributes_attachment['adoption'][node] = 0
            self.node_attributes_attachment['num_neighbor_adopted'][node] = 0
            self.node_attributes_attachment['adoption_time'][node] = -1
        
        if self.start > 0:
            nodes_list = list(range(self.current_node_number))
            np.random.shuffle(nodes_list)
            for node in nodes_list:
                    name = str(self.node_attributes_attachment[self.seed_type][node])
                    try:
                        if self.seed_node[name]>0:
                            self.node_attributes_attachment['adoption_time'][node] = 0
                            self.node_attributes_attachment['adoption'][node] = 1
                            self.seed_node[name] = self.seed_node[name] - 1
                    except:
                        continue
            self.update_num_neighbor_adopted()

        if self.start < 0:
            nodes_list = list(range(self.current_node_number))
            np.random.shuffle(nodes_list)
            for node in nodes_list:
                    name = str(self.node_attributes_attachment[self.seed_type][node])
                    try:
                        if self.seed_node[name]>0:
                            self.node_attributes_attachment['adoption_time'][node] = 0
                            self.node_attributes_attachment['adoption'][node] = 1
                            self.seed_node[name] = self.seed_node[name] - 1
                    except:
                        continue        
            
            df = self.cal_class_degree()
            if self.class_focus == 'random':
                seed_id, seed_degree = [],[]
                for node,degree in zip(df['id'].values,df['middle'].values):
                    if self.node_attributes_attachment['adoption'][node] != 1:
                        seed_id.append(node)
                        seed_degree.append(degree)
                df = pd.DataFrame({'id':seed_id, 'degree':seed_degree})
                df = df.sample(frac=1).reset_index(drop=True)

            else:
                seed_id, seed_degree = [],[]
                for node,degree in zip(df['id'].values,df[self.class_focus].values):
                    if self.node_attributes_attachment['adoption'][node] != 1:
                        seed_id.append(node)
                        seed_degree.append(degree)
                df = pd.DataFrame({'id':seed_id, 'degree':seed_degree}).sort_values(by=['degree'],ascending=False)

            n_seed = abs(self.start)
            nodes_list = df['id'].values[:int(n_seed)]
            for node in nodes_list:
                self.node_attributes_attachment['adoption_time'][node] = 0
                self.node_attributes_attachment['adoption'][node] = 1
            self.update_num_neighbor_adopted()

    def build_network(self):
        self.generate_nodes()
        self.generate_edges()
        self.set_node_degree()
        self.create_seed()
        self.reset()


