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

# -------------------------------------------------------------------
# Utility: run a function over a list of args using multiprocessing.
# Returns the list of results from func(args_i) for each args_i.
# -------------------------------------------------------------------
def parallelize_dataframe(args, func, n_cores):
    pool = Pool(n_cores)
    results = pool.map(func,args)
    pool.close()
    pool.join()
    return results

# -------------------------------------------------------------------
# Per-batch county data preparation:
# - Slices global `counties` for this batch_id
# - For each county, run NetworkPrep.createData() to produce:
#   demo_data.csv, tract_coord.csv, M.npy, curves, benchmarks, etc.
# -------------------------------------------------------------------
def calPrep(batch_id):
    counties_batch = counties[batch_id*batch_size:(batch_id+1)*batch_size]
    for county in counties_batch:
        nwp_obj = nwp.NetworkPrep(state, county, emp)
        nwp_obj.createData()
        del nwp_obj

# -------------------------------------------------------------------
# Per-batch diffusion calibration:
# - Builds parameter vector `par` (homophily, distance exponent, etc.)
# - Sets `anna` experiment label by state + county + repetition id
# - Computes number of modeled nodes by dividing population by
#   `people_per_node`
# - For reasonably-sized counties (<= 200,000 modeled nodes),
#   runs:
#     1) npm.NetworkParameter.calIntialPQ() to estimate seeds
#     2) npm.NetworkParameter.calFinalPQ() to fit final p,q via search
# -------------------------------------------------------------------
def calDiffusion(batch_id):
    counties_batch = counties[batch_id*batch_size:(batch_id+1)*batch_size]
    for county in counties_batch:
        par = [-1, expr, 3, 3, 0, 13,'None']  # [homo, r_exp, k_exp, k_min, start, end, class_focus]
        anna = 'state_'+''.join(i.lower() for i in county.split())+'_'+str(simi)
        number_node = int(pd.read_csv(os.path.realpath(os.path.join('..', 'data', state, county, 'demo_data.csv')),converters={'GEOID': str}).POPULATION.sum()/people_per_node)
        if number_node <= 200000:
            npm_obj = npm.NetworkParameter(number_node, par, state, county, rep_num, anna)
            npm_obj.calIntialPQ()
            npm_obj.calFinalPQ()
            del npm_obj

# -------------------------------------------------------------------
# Global configuration:
# - people_per_node: population downscaling factor
# - rep_num: number of ABM repetitions per search point
# - expr: distance exponent used in network generation
# -------------------------------------------------------------------
people_per_node = 5; rep_num = 10; expr = 1

# ================================================================================================
# Washington run : If you don't have core less than 39, please reduce run_core.
# ================================================================================================
state = 'wa'; run_core = 39
# Load county list for WA
state_county = pd.read_csv(os.path.join('..','data', state, 'fips-by-state.csv'),encoding= 'unicode_escape')
state_county = state_county[state_county['state']=='WA']
counties = state_county.name.str.split(' County',expand=True)[0].unique()
batch_size = int(np.ceil(len(counties)/run_core))

# Load WA empirical registration data (DOL)
emp = pd.read_csv(os.path.join('..','data', state,'wa_new_ev_registrations.csv.gz'))
emp = emp.rename(columns={'DOL Transaction Date':'date','2020 Census Tract':'tract','County':'county'})

# Prepare data for all WA counties in parallel
results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calPrep,n_cores=run_core)

# Run diffusion calibration multiple times (simi = 0..3)
for simi in range(4):
    results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calDiffusion,n_cores=run_core)

# ================================================================================================
# California run : If you don't have core less than 58, please reduce run_core.
# ================================================================================================
state = 'ca'; run_core = 58
# Load county list for CA
state_county = pd.read_csv(os.path.join('..','data', state, 'fips-by-state.csv'),encoding= 'unicode_escape')
state_county = state_county[state_county['state']=='CA']
counties = state_county.name.str.split(' County',expand=True)[0].unique()
batch_size = int(np.ceil(len(counties)/run_core))

# Load CA empirical registration data (CVRP)
emp = pd.read_excel(os.path.join('..','data', state,'CVRPStats.xlsx'))
emp = emp.rename(columns={'Application Date':'date','Census Tract':'tract','County':'county'})

# Prepare data for all CA counties in parallel
results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calPrep,n_cores=run_core)

# Run diffusion calibration multiple times (simi = 0..3)
for simi in range(4):
    results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calDiffusion,n_cores=run_core)

