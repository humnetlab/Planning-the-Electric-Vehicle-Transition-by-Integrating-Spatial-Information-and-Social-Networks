import NetworkParameter as npm
import NetworkPrep as nwp
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

def parallelize_dataframe(args, func, n_cores):
    pool = Pool(n_cores)
    results = pool.map(func,args)
    pool.close()
    pool.join()
    return results

def calPrep(batch_id):
    counties_batch = counties[batch_id*batch_size:(batch_id+1)*batch_size]
    for county in counties_batch:
        nwp_obj = nwp.NetworkPrep(state, county, emp)
        nwp_obj.createData()
        del nwp_obj

def calDiffusion(batch_id):
    counties_batch = counties[batch_id*batch_size:(batch_id+1)*batch_size]
    for county in counties_batch:
        par = [-1, expr, 3, 3, 0, 13,'None']
        anna = 'state_'+''.join(i.lower() for i in county.split())+'_'+str(simi)
        number_node = int(pd.read_csv(os.path.realpath(os.path.join('..', 'data', state, county, 'demo_data.csv')),converters={'GEOID': str}).POPULATION.sum()/people_per_node)
        npm_obj = npm.NetworkParameter(number_node, par, state, county, rep_num, anna)
        npm_obj.calIntialPQ()
        npm_obj.calFinalPQ()
        del npm_obj

people_per_node = 5; rep_num = 10; expr = 2

state = 'wa'; run_core = 39
state_county = pd.read_csv(os.path.join('..','data', state, 'fips-by-state.csv'),encoding= 'unicode_escape')
state_county = state_county[state_county['state']=='WA']
counties = state_county.name.str.split(' County',expand=True)[0].unique()
batch_size = int(np.ceil(len(counties)/run_core))

emp = pd.read_csv(os.path.join('..','data', state,'wa_new_ev_registrations.csv.gz'))
emp = emp.rename(columns={'DOL Transaction Date':'date','2020 Census Tract':'tract','County':'county'})
results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calPrep,n_cores=run_core)

for simi in range(5):
    results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calDiffusion,n_cores=run_core)

state = 'ca'; run_core = 58
state_county = pd.read_csv(os.path.join('..','data', state, 'fips-by-state.csv'),encoding= 'unicode_escape')
state_county = state_county[state_county['state']=='CA']
counties = state_county.name.str.split(' County',expand=True)[0].unique()
batch_size = int(np.ceil(len(counties)/run_core))

emp = pd.read_excel(os.path.join('..','data', state,'CVRPStats.xlsx'))
emp = emp.rename(columns={'Application Date':'date','Census Tract':'tract','County':'county'})

for simi in range(5):
    results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calDiffusion,n_cores=run_core)


