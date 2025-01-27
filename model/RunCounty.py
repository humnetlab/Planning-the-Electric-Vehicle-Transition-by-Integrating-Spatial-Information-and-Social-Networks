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

def calDiffusion(batch_id):
    expr = exprs[batch_id%3]
    county = 'Los Angeles'
    par = [-1, expr, 3, 3, 0, 13,'None']
    anna = ''.join(i.lower() for i  in county.split())+'_'+str(simi)
    number_node = int(pd.read_csv(os.path.realpath(os.path.join('..', 'data', state, county, 'demo_data.csv')),converters={'GEOID': str}).POPULATION.sum()/people_per_node)
    npm_obj = npm.NetworkParameter(number_node, par, state, county, rep_num, anna)
    npm_obj.calIntialPQ()
    npm_obj.calFinalPQ()
    del npm_obj

rep_num = 5; run_core = 3; state = 'ca'
people_per_node = 5; exprs = [1,3,10]

for simi in range(5):
    results = parallelize_dataframe([batch_id for batch_id in range(run_core)],calDiffusion,n_cores=run_core)

