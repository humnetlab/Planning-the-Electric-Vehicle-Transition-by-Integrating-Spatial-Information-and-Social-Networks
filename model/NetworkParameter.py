import networkx as nx
import numpy as np
import os
import networkit as nk
import pandas as pd
import pickle
import warnings
from itertools import product
from SearchParameter import *
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')


def residual(var, t, y, m):
    p = var[0]
    q = var[1]
    A = 1/p
    Bass = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))  
    if (p<0)or(p>1)or(q<0)or(q>1):
        Penalization = 1e30*(np.abs(min(0,p))+max(0,p-1)+np.abs(min(0,q))+max(0,q-1))
        return np.abs(Bass - y) + Penalization
    else:
        return np.abs(Bass - y)

def bassfit(x,y,m):
    vars = [1e-4,1e-1]
    varfinal,success = leastsq(residual, vars, args=(x, y, m))
    p = varfinal[0]; q = varfinal[1]
    return p, q
    
class NetworkParameter:
    def __init__(self, number_node, par, state, county, rep_num, anna):
        self.state = state
        self.county = county
        self.number_node = number_node
        self.par = par
        self.homo = par[0]
        self.r_exp = par[1]
        self.k_exp = par[2]
        self.k_min = par[3]
        self.end = par[5]
        self.rep_num = rep_num
        self.anna = anna
        self.emp_seed_group = pickle.load(open(os.path.realpath(os.path.join( '..', 'data', self.state, self.county,'emp_seed_group.pkl')), 'rb'), encoding='bytes')
        self.emp_pop_group = pickle.load(open(os.path.realpath(os.path.join( '..', 'data', self.state, self.county,'emp_pop_group.pkl')), 'rb'), encoding='bytes')
        self.scale = self.number_node/pd.read_csv(os.path.realpath(os.path.join('..', 'data', self.state, self.county,'demo_data.csv')),converters={'GEOID': str}).fillna(0).POPULATION.sum()
        self.demo = pd.read_csv(os.path.realpath(os.path.join('..', 'data', self.state, self.county,'demo_data.csv')),converters={'GEOID': str}).fillna(0)
        self.RESULT_PATH = os.path.join('..', 'result', self.state, self.county)

        
    def calIntialPQ(self):
        emp_seed_group = self.emp_seed_group
        emp_pop_group = self.emp_pop_group
        
        m_low = self.demo[self.demo['CLASS']=='Low'].POPULATION.sum()
        m_mid = self.demo[self.demo['CLASS']=='Middle'].POPULATION.sum()
        m_high = self.demo[self.demo['CLASS']=='High'].POPULATION.sum()

        if m_low>0:
            p_abmini_values_low = np.linspace(1e-4,1e-3,3); q_abmini_values_low = np.linspace(1e-3,1e-2,3)
        else:
            p_abmini_values_low = np.array([-1]); q_abmini_values_low = np.array([-1])
        if m_mid>0:
            p_abmini_values_mid = np.linspace(1e-4,1e-3,3); q_abmini_values_mid = np.linspace(1e-3,1e-2,3)
        else:
            p_abmini_values_mid = np.array([-1]); q_abmini_values_mid = np.array([-1])
        if m_high>0:
            p_abmini_values_high = np.linspace(1e-4,1e-3,3); q_abmini_values_high = np.linspace(1e-3,1e-2,3)
        else:
            p_abmini_values_high = np.array([-1]); q_abmini_values_high = np.array([-1])

        pq_values = []
        somelist = [p_abmini_values_low,q_abmini_values_low,p_abmini_values_mid,q_abmini_values_mid,p_abmini_values_high,q_abmini_values_high]
        for element in product(*somelist):
            pq_values.append(element)
    
        p_bass_values_high = []; q_bass_values_high = []; p_abm_values_high = []; q_abm_values_high = []
        p_bass_values_mid = []; q_bass_values_mid = []; p_abm_values_mid = []; q_abm_values_mid = []
        p_bass_values_low = []; q_bass_values_low = []; p_abm_values_low = []; q_abm_values_low = []
        
        count = 0; number_node = min(10000,self.number_node); rep_num = 10; start = 0
        scale = number_node/pd.read_csv(os.path.realpath(os.path.join('..', 'data', self.state, self.county,'demo_data.csv')),converters={'GEOID': str}).POPULATION.sum()
        func_name = 'initial_pq_'+self.anna+'_'+str(number_node)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)
        
        for p_value_low,q_value_low,p_value_mid,q_value_mid,p_value_high,q_value_high in pq_values:
            p_value_seed = min([np.abs(p_value_low),np.abs(p_value_mid),np.abs(p_value_high)]); q_value_seed = min([np.abs(q_value_low),np.abs(q_value_mid),np.abs(q_value_high)])
            t=np.array(range(100))
            A = 1/p_value_seed; bass = (1-A*(np.exp(-(p_value_seed+q_value_seed)*t)*p_value_seed+np.exp(-(p_value_seed+q_value_seed)*t)*q_value_seed)/(1+A*q_value_seed*np.exp(-(p_value_seed+q_value_seed)*t)))  
            end = np.argmax(bass[0+1:100]-bass[0:100-1])+2; t = np.array(range(start,end))

            p_value_search = [[p_value_low],[p_value_mid],[p_value_high]]
            q_value_search = [[q_value_low],[q_value_mid],[q_value_high]]

            par_ini = self.par.copy(); par_ini[4] = start; par_ini[5] = end
            spmob = SearchParameter(p_value_search, q_value_search, number_node, par_ini, self.state, self.county, rep_num, 0, os.path.join(func_name, str(count)))
            spmob.randomsearch()
            if count == 0:
                cc = nk.components.ConnectedComponents(spmob.G)
                cc.run()
                assert(cc.numberOfComponents()==1)
            del spmob
        
            low_curve = []; mid_curve = []; high_curve = []
            for j in range(rep_num):
                sub_tract = (pd.read_csv(os.path.join(self.RESULT_PATH, os.path.join(func_name, str(count)), 'curves', 'tract_curve_'+str(j)+'.csv'),index_col=0).cumsum(axis=0).transpose()/scale)
                low_curve.append(emp_pop_group[0].merge(sub_tract,left_on='GEOID',right_on=sub_tract.index)[range(end)].sum())
                mid_curve.append(emp_pop_group[1].merge(sub_tract,left_on='GEOID',right_on=sub_tract.index)[range(end)].sum())
                high_curve.append(emp_pop_group[2].merge(sub_tract,left_on='GEOID',right_on=sub_tract.index)[range(end)].sum())
            
            if m_low>0:
                p_bass_low,q_bass_low = bassfit(t,np.mean(low_curve,axis=0)[start:end],emp_pop_group[0]['POPULATION'].sum())
            else:
                p_bass_low,q_bass_low = -1,-1
            if m_mid>0:
                p_bass_mid,q_bass_mid = bassfit(t,np.mean(mid_curve,axis=0)[start:end],emp_pop_group[1]['POPULATION'].sum())
            else:
                p_bass_mid,q_bass_mid = -1,-1
            if m_high>0:
                p_bass_high,q_bass_high = bassfit(t,np.mean(high_curve,axis=0)[start:end],emp_pop_group[2]['POPULATION'].sum())
            else:
                p_bass_high,q_bass_high = -1,-1
        
            if max(q_bass_low,q_bass_mid,q_bass_high)<0.1:
                p_bass_values_low.append(p_bass_low)
                q_bass_values_low.append(q_bass_low)
                p_abm_values_low.append(p_value_low)
                q_abm_values_low.append(q_value_low)
                p_bass_values_mid.append(p_bass_mid)
                q_bass_values_mid.append(q_bass_mid)
                p_abm_values_mid.append(p_value_mid)
                q_abm_values_mid.append(q_value_mid)
                p_bass_values_high.append(p_bass_high)
                q_bass_values_high.append(q_bass_high)
                p_abm_values_high.append(p_value_high)
                q_abm_values_high.append(q_value_high)
            count = count+1

        X = []; y=[]
        if m_low>0:
            X += [p_bass_values_low,q_bass_values_low]
            y += [p_abm_values_low,q_abm_values_low]
        if m_mid>0:
            X += [p_bass_values_mid,q_bass_values_mid]
            y += [p_abm_values_mid,q_abm_values_mid]
        if m_high>0:
            X += [p_bass_values_high,q_bass_values_high]
            y += [p_abm_values_high,q_abm_values_high]
        X = np.array(X).transpose()
        y = np.array(y).transpose()
        reg = LinearRegression(fit_intercept=True).fit(X, y)

        print('fit score', reg.score(X, y))
        print('bass seed', [emp_seed_group[0]+emp_seed_group[1]+emp_seed_group[2]])
        
        X_prime = []
        if m_low>0:
            X_prime += emp_seed_group[0]
        if m_mid>0:
            X_prime += emp_seed_group[1]
        if m_high>0:
            X_prime += emp_seed_group[2]
        y_prime = reg.predict([X_prime])[0]
        
        count = 0; y_p=[]
        if m_low>0:
            y_p += [y_prime[count],y_prime[count+1]]
            count = count + 2
        else:
            y_p += [-1,-1]
        if m_mid>0:
            y_p += [y_prime[count],y_prime[count+1]]
            count = count + 2
        else:
            y_p += [-1,-1]
        if m_high>0:
            y_p += [y_prime[count],y_prime[count+1]]
            count = count + 2
        else:
            y_p += [-1,-1]
        print('initial seed', y_p)
        np.save(os.path.join(self.RESULT_PATH, func_name, 'sim_seed_group.npy'),np.array(y_p))


    def calFinalPQ(self):
        p_value_seed_low, q_value_seed_low, p_value_seed_mid, q_value_seed_mid, p_value_seed_high, q_value_seed_high = np.load(os.path.join(self.RESULT_PATH, os.path.join('initial_pq_'+self.anna+'_'+str(min(10000,self.number_node))+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)), 'sim_seed_group.npy'))
        p_value_search = [[p_value_seed_low],[p_value_seed_mid],[p_value_seed_high]]
        q_value_search = [[q_value_seed_low],[q_value_seed_mid],[q_value_seed_high]]
        func_name = 'fit_pq_'+self.anna+'_'+str(self.number_node)+'_'+str(self.homo)+'_'+str(self.r_exp)+'_'+str(self.k_exp)+'_'+str(self.k_min)
        spmob = SearchParameter(p_value_search, q_value_search, self.number_node, self.par, self.state, self.county, self.rep_num, 1, func_name)
        spmob.randomsearch()
