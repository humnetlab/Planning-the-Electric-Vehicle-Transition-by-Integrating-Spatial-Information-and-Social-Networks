from NetworkCreators import *
from SimulateParameter import *
import numpy as np
import networkit as nk
import pickle
import os
from itertools import product
import matplotlib.pyplot as plt
import powerlaw


class SearchParameter:
    def __init__(self, p_value, q_value, number_node, par, state, county, rep_num, best, func_name):
        self.p_value = p_value
        self.q_value = q_value
        self.end = par[5]
        self.number_node = number_node
        self.par = par
        self.state = state
        self.county = county
        self.rep_num = rep_num
        self.best = best
        self.func_name = func_name
        self.RESULT_PATH = os.path.join('..', 'result', self.state, self.county)
        if par[4]<0:
            self.start = 12
        else:
            self.start = par[4]

    def experiment(self,p_value,q_value,start,end):
        simulation_paras = {"p_value": p_value,
                            "q_value": q_value}
        simulation = SimulateParameter(self.G, simulation_paras, start, end, self.state, self.county)
        simulation.run()
        if self.best == 0:
            group_error = 0
            state_curve = simulation.output_curve_by_state()
            tract_curve = simulation.output_curve_by_tract()
            agent_curve = simulation.output_curve_by_agent()
        else:
            group_error = simulation.calculate_absolute_error_group()
            state_curve = []
            tract_curve = []
            agent_curve = []
        return group_error, state_curve, tract_curve, agent_curve

    def randomsearch(self):
        group_num = 3; max_iter = 100; max_search = 50; min_error = 1e12; 
        # create result directory
        subfolder_name = ['parameters','metrics','curves']

        for subfolder in subfolder_name:
            new_path =  os.path.join( '..','result', self.state, self.county, self.func_name, subfolder)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
        
        # regression model
        if self.best == 0:
            self.G = NetworkCreators(0,self.par+[self.number_node,self.state,self.county])
            self.G.build_network()
            
            cc = nk.components.ConnectedComponents(self.G)
            cc.run()
            print("number of components = ", cc.numberOfComponents(),",size of biggest component = ", len(cc.getComponents()[0]))
            dd = sorted(nk.centrality.DegreeCentrality(self.G).run().scores(), reverse=True)
            print('# nodes, # edges = ', self.G.numberOfNodes(), self.G.numberOfEdges())

            p_value_sample = []; q_value_sample =[]
            for group_idx in range(group_num):
                p_value_sample.append(self.p_value[group_idx][0])
                q_value_sample.append(self.q_value[group_idx][0])
            
            for j in range(self.rep_num):
                pd_error, state_curve, tract_curve, agent_curve = self.experiment(p_value_sample, q_value_sample, self.start, self.end)
                with open(os.path.join(self.RESULT_PATH, self.func_name, 'curves', 'state_curve_'+str(j)+'.pkl'), "wb") as f:
                    pickle.dump(state_curve, f)
                tract_curve.to_csv(os.path.join(self.RESULT_PATH, self.func_name, 'curves', 'tract_curve_'+str(j)+'.csv'))
                agent_curve.to_csv(os.path.join(self.RESULT_PATH, self.func_name, 'curves', 'agent_curve_'+str(j)+'.csv'))
                self.G.reset()

        # best model
        if self.best == 1:
            self.G = NetworkCreators(0,self.par+[self.number_node,self.state,self.county])
            self.G.build_network()
            cc = nk.components.ConnectedComponents(self.G)
            cc.run()
            assert(cc.numberOfComponents()==1)
            
            pq_searched = []; noprogress_count = 0; delta_p = 0.9; delta_q = 0.9; rec = 0.9
            demo_data = pd.read_csv(os.path.realpath(os.path.join('..', 'data', self.state, self.county, 'demo_data.csv')),converters={'GEOID': str})
            tract_emp = (pd.read_csv(os.path.join( '..', 'data', self.state, self.county, 'tract_curve.csv'),converters={'tract': str}))
            tract_emp = tract_emp[tract_emp['year'] == 6]

            tract_emp_demo = tract_emp.merge(demo_data,left_on='tract',right_on='GEOID')
            tract_high = tract_emp_demo[(tract_emp_demo['cum_reg']>0)&(tract_emp_demo['CLASS']=='High')]
            tract_mid = tract_emp_demo[(tract_emp_demo['cum_reg']>0)&(tract_emp_demo['CLASS']=='Middle')]
            tract_low = tract_emp_demo[(tract_emp_demo['cum_reg']>0)&(tract_emp_demo['CLASS']=='Low')]
            
            m_low = demo_data[demo_data['CLASS']=='Low'].POPULATION.sum()
            m_mid = demo_data[demo_data['CLASS']=='Middle'].POPULATION.sum()
            m_high = demo_data[demo_data['CLASS']=='High'].POPULATION.sum()

            if m_low > 0:
                p_low_min = tract_low['cum_reg'].sum()/m_low
            else:
                p_low_min = -1

            if m_mid > 0:
                p_mid_min = tract_mid['cum_reg'].sum()/m_mid
            else:
                p_mid_min = -1

            if m_high > 0:
                p_high_min = tract_high['cum_reg'].sum()/m_high
            else:
                p_high_min = -1
            
            for i in range(max_iter): 
                if noprogress_count > 50:
                    print('==================================== Early Stop ====================================')
                    break
                else:
                    print('====================================','Iteration ', str(i),'====================================')
                    pd_index = 0
                    pq_values = []; somelists = []; pq_indexs = []; p_value_samples = []; q_value_samples = []; pq_errors = []
                    
                    for group_idx in range(group_num):
                        if i == 0:
                            p_value_center =  min(max(self.p_value[group_idx][0],1e-6),1)
                        else:
                            p_value_center = min(max(p_value_star[group_idx],1e-6),1)
                        if (group_idx == 0):
                            if (m_low == 0):
                                somelists.append([-1])
                            else:
                                somelists.append([min(max(p_value_center*(1-delta_p),1e-6),p_low_min),min(p_value_center,p_low_min),min(min(p_value_center*(1+delta_p),1),p_low_min)])
                        if (group_idx == 1):
                            if (m_mid == 0):
                                somelists.append([-1])
                            else:                        
                                somelists.append([min(max(p_value_center*(1-delta_p),1e-6),p_mid_min),min(p_value_center,p_mid_min),min(min(p_value_center*(1+delta_p),1),p_mid_min)])
                        if (group_idx == 2):
                            if (m_high == 0):
                                somelists.append([-1])
                            else:
                                somelists.append([min(max(p_value_center*(1-delta_p),1e-6),p_high_min),min(p_value_center,p_high_min),min(min(p_value_center*(1+delta_p),1),p_high_min)])      

                    for group_idx in range(group_num):
                        if i == 0:
                            q_value_center = min(max(self.q_value[group_idx][0],1e-6),1)
                        else:
                            q_value_center = min(max(q_value_star[group_idx],1e-6),1)
                        if (group_idx == 0):
                            if (m_low == 0):
                                somelists.append([-1])
                            else:
                                somelists.append([max(q_value_center*(1-delta_q),1e-6),q_value_center,min(q_value_center*(1+delta_q),1)])
                        if (group_idx == 1):
                            if (m_mid == 0):
                                somelists.append([-1])
                            else:
                                somelists.append([max(q_value_center*(1-delta_q),1e-6),q_value_center,min(q_value_center*(1+delta_q),1)])
                        if (group_idx == 2):
                            if (m_high == 0):
                                somelists.append([-1])
                            else:
                                somelists.append([max(q_value_center*(1-delta_q),1e-6),q_value_center,min(q_value_center*(1+delta_q),1)])
                        
                    for element in product(*somelists):
                        pq_values.append(element)
                    
                    for k in range(max_search):
                        pq_value = pq_values[np.random.choice(range(len(pq_values)),1)[0]]; 
                        if pq_value not in pq_searched:
                            p_value_sample = pq_value[:group_num]
                            q_value_sample = pq_value[group_num:]
                            for j in range(self.rep_num):
                                pd_error, state_curve, tract_curve, agent_curve = self.experiment(p_value_sample, q_value_sample, self.start ,self.end)
                                pq_indexs.append(pd_index);p_value_samples.append(p_value_sample);q_value_samples.append(q_value_sample);pq_errors.append(pd_error)
                                pq_searched.append(pq_value)
                                self.G.reset()
                            pd_index += 1
                    
                    if len(pq_indexs)>0:
                        data = pd.DataFrame([pq_indexs,p_value_samples,q_value_samples,pq_errors],index=['idx','p','q','error']).transpose()
                        data_error = pd.pivot_table(data, values=['error'], index=['idx'], aggfunc=[np.mean,np.std]).reset_index().sort_values(by=[('mean','error')])
                        p_value_min = data[data['idx'] == data_error['idx'].values[0]]['p'].values[0]
                        q_value_min = data[data['idx'] == data_error['idx'].values[0]]['q'].values[0]
                        error_star = data[data['idx'] == data_error['idx'].values[0]]['error'].values[0]

                        if error_star<min_error:
                            p_value_star = p_value_min; q_value_star = q_value_min; min_error = error_star
                            delta_p = 0.9; delta_q = 0.9;  noprogress_count = 0
                            print('so far optimal:', p_value_star,q_value_star,error_star)
                        else:
                            noprogress_count = noprogress_count + 1
                            delta_p = delta_p * rec
                            delta_q = delta_q * rec
            
            for j in range(10):
                simulation_paras = {"p_value": p_value_star,
                                    "q_value": q_value_star}
                simulation = SimulateParameter(self.G, simulation_paras, self.start, 200, self.state, self.county)
                simulation.run()
                state_curve = simulation.output_curve_by_state()
                tract_curve = simulation.output_curve_by_tract()
                agent_curve = simulation.output_curve_by_agent()
                self.G.reset()

                with open(os.path.join(self.RESULT_PATH, self.func_name, 'curves', 'state_curve_'+str(j)+'.pkl'), "wb") as f:
                    pickle.dump(state_curve, f)
                tract_curve.to_csv(os.path.join(self.RESULT_PATH, self.func_name, 'curves', 'tract_curve_'+str(j)+'.csv'))
                agent_curve.to_csv(os.path.join(self.RESULT_PATH, self.func_name, 'curves', 'agent_curve_'+str(j)+'.csv'))

            data.to_csv(os.path.join(self.RESULT_PATH, self.func_name,'metrics','data_error.csv'))
            with open(os.path.join(self.RESULT_PATH, self.func_name, 'parameters', 'p_value_star.pkl'), "wb") as f:
                pickle.dump(p_value_star, f)
            with open(os.path.join(self.RESULT_PATH, self.func_name, 'parameters', 'q_value_star.pkl'), "wb") as f:
                pickle.dump(q_value_star, f)
