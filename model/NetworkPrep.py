import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import os
from geopy.distance import great_circle
import geopandas as gpd
from scipy.optimize import leastsq
from scipy.optimize import least_squares
import pickle
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Residual function for Bass diffusion least-squares fitting
# Used in optimization to find best p, q parameters.
# Penalizes values of p or q that are outside the valid [0,1] range.
# -------------------------------------------------------------------
def residual(var, t, y, m):
    p = var[0]
    q = var[1]
    A = 1/p
    Bass = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))  
    if (p<0)or(p>1)or(q<0)or(q>1):
        Penalization = 1e30*(min(0,p)+max(0,p-1)+np.abs(min(0,q))+max(0,q-1))
        return np.abs(Bass - y) + Penalization
    else:
        return np.abs(Bass - y)
    
# -------------------------------------------------------------------
# Fit Bass diffusion parameters p (innovation) and q (imitation)
# from observed adoption data using least squares.
# -------------------------------------------------------------------
def bassfit(x,y,m):
    vars = [1e-4,1e-1]  # initial guess for [p, q]
    varfinal,success = leastsq(residual, vars, args=(x, y, m))
    p = varfinal[0]; q = varfinal[1]
    return p, q

# -------------------------------------------------------------------
# Group adoption records into yearly data.
# Produces yearly cumulative and incremental registrations
# for a given tract/county key and value.
# -------------------------------------------------------------------
def year_groupby(df, key = '', value=''):
    day_unit = 365
    df = df[df[key] == value]
    df = df[['date']]
    df['sale'] = 1
    # Compute year index relative to baseline date (2010-02-16)
    df['year'] = ((df['date']-pd.to_datetime('2010-02-16 00:00:00'))//day_unit).dt.days

    # Add missing daily records with zero sales to ensure continuity
    df_add = pd.DataFrame()
    df_add['date'] = pd.date_range('2010-02-16 00:00:00', '2023-01-31 23:58:00', freq='D')
    df_add['sale'] = 0
    df_add['year'] = ((df_add['date']-pd.to_datetime('2010-02-16 00:00:00'))//day_unit).dt.days

    df = pd.concat([df,df_add])

    # Sort and compute cumulative/differential registrations
    df_year = df.sort_values(by=['year'])
    df_year.rename(
        {'sale': 'new_reg'}, inplace=True, axis=1
    )
    df_year['cum_reg'] = df_year['new_reg'].cumsum()
    df_year = df_year.drop_duplicates(subset=['year'],keep='last')
    df_year['dif_reg'] = np.hstack([df_year['cum_reg'].values[0],df_year['cum_reg'].values[1:]-df_year['cum_reg'].values[:-1]])
    # Ensure GEOID/tract formatting (leading zero padding if needed)
    if key == 'tract' and len(str(int(value)))<11:
        df_year[key] = '0'+str(value)
    else:
        try:
            df_year[key] = str(int(value))
        except:
            df_year[key] = str(value)

    return df_year

# -------------------------------------------------------------------
# NetworkPrep class
# Prepares demographic, spatial, and adoption curve data
# for county/state-level diffusion network modeling.
# -------------------------------------------------------------------
class NetworkPrep:
    def __init__(self, state_file, county_file, emp):
        self.emp = emp
        self.county_name = county_file #.replace(" ", "_").lower()
        self.state_file = state_file
        self.state_name = self.state_file.upper()
        self.DATA_PATH = os.path.join('..','data',state_file,self.county_name)
        # Load state-specific tract shapefiles
        if state_file == 'ca':
            self.tract_shp = gpd.read_file(os.path.join('..','data',state_file,'tract','tl_2019_06_tract.shp'))
        if state_file == 'wa':
            self.tract_shp = gpd.read_file(os.path.join('..','data',state_file,'tract','tl_2020_53_tract.shp'))
        # Create county data folder if not exists
        if not os.path.exists(self.DATA_PATH):
            os.makedirs(self.DATA_PATH)
    
    # ---------------------------------------------------------------
    # Create county-specific demographic dataset and coordinates file.
    # Saves demo_data.csv and tract_coord.csv for county tracts.
    # ---------------------------------------------------------------
    def createDemo(self):
        demo_tract = pd.read_csv(os.path.join('..','data',self.state_file,'demo_data.csv'),converters={'GEOID': str}).copy()
        demo_tract = demo_tract[demo_tract['COUNTY']==self.county_name]
        demo_tract.to_csv(os.path.join(self.DATA_PATH, 'demo_data.csv'),index=False)

        tract_shp = self.tract_shp
        tract_shp = tract_shp[tract_shp['GEOID'].isin(demo_tract['GEOID'].values)]
        # Add geometric features: centroid longitude/latitude, area
        tract_shp['Longitude'] = tract_shp.geometry.centroid.x
        tract_shp['Latitude'] = tract_shp.geometry.centroid.y
        tract_shp['Area'] = tract_shp.geometry.to_crs({'proj':'cea'}).area/10**6
        tract_coord = tract_shp
        tract_coord.to_csv(os.path.join(self.DATA_PATH,'tract_coord.csv'),index=False)

    # ---------------------------------------------------------------
    # Create adoption curves for tracts, county, and state.
    # Fits Bass model seeds (p,q) for low/mid/high income classes.
    # Saves curves and pickle files for later modeling.
    # ---------------------------------------------------------------
    def createCurve(self):
        demo_tract = pd.read_csv(os.path.join(self.DATA_PATH, 'demo_data.csv'),converters={'GEOID': str})
        emp = self.emp.copy()
        emp = emp[emp['county']==self.county_name]
        emp['date'] = pd.to_datetime(emp['date'])
        
        key ='tract'
        emp_year = []
        value_list = emp[key].dropna().unique()
        # Aggregate yearly adoption for each tract
        for value in value_list:
            emp_grouped_year = year_groupby(emp,key,value)
            emp_grouped_year = emp_grouped_year[[key,'year','dif_reg','cum_reg']]
            emp_year.append(emp_grouped_year)
        emp_year_key = pd.concat(emp_year)
        emp_year_key = emp_year_key[emp_year_key['tract'].isin(demo_tract.GEOID.values)]
        emp_year_key.to_csv(os.path.join(self.DATA_PATH, key+'_curve.csv'),index=False)
        
        # County-level aggregate curve
        emp_year_county = pd.pivot_table(emp_year_key, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        emp_year_county['county'] = self.county_name
        emp_year_county.to_csv(os.path.join(self.DATA_PATH, 'county_curve.csv'),index=False)

        # State-level aggregate curve
        emp_year_state = pd.pivot_table(emp_year_key, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        emp_year_state.to_csv(os.path.join(self.DATA_PATH, 'state_curve.csv'),index=False)
        
        # Year mapping file (year index -> actual year)
        year_date = pd.DataFrame()
        year_date['date'] = np.array(range(13))
        year_date['year'] = np.array(range(2010,2023))
        year_date.to_csv(os.path.join(self.DATA_PATH, 'year_date.csv'),index=False)
        
        # Merge with demo data to stratify by income class
        tract_curve = pd.read_csv(os.path.join(self.DATA_PATH, 'tract_curve.csv'),converters={'tract': str})
        demo_tract = pd.read_csv(os.path.join(self.DATA_PATH, 'demo_data.csv'),converters={'GEOID': str})
        
        end = 13
        demo_vis = tract_curve.merge(demo_tract,left_on='tract',right_on='GEOID',how='left')#.fillna(0)
        ea_low = demo_vis[demo_vis['CLASS'] == 'Low']
        ea_mid = demo_vis[demo_vis['CLASS'] == 'Middle']
        ea_high = demo_vis[demo_vis['CLASS'] == 'High']

        pop_low = demo_tract[demo_tract['CLASS']  == 'Low']
        pop_mid = demo_tract[demo_tract['CLASS'] == 'Middle']
        pop_high = demo_tract[demo_tract['CLASS']  == 'High']

        # County-level adoption curves by class
        low_county_curve = pd.pivot_table(ea_low, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        mid_county_curve = pd.pivot_table(ea_mid, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        high_county_curve = pd.pivot_table(ea_high, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        
        # Handle missing classes by filling with zero
        if len(low_county_curve)==0:
            low_county_curve = pd.DataFrame({'year':list(range(end)),'cum_reg':np.zeros(end),'dif_reg':np.zeros(end)})
        if len(mid_county_curve)==0:
            mid_county_curve = pd.DataFrame({'year':list(range(end)),'cum_reg':np.zeros(end),'dif_reg':np.zeros(end)})
        if len(high_county_curve)==0:
            high_county_curve = pd.DataFrame({'year':list(range(end)),'cum_reg':np.zeros(end),'dif_reg':np.zeros(end)})

        # Save population groups to pickle
        emp_pop_group = {}
        emp_pop_group[0] = pop_low
        emp_pop_group[1] = pop_mid
        emp_pop_group[2] = pop_high
        with open(os.path.realpath(os.path.join(self.DATA_PATH, 'emp_pop_group.pkl')), 'wb') as f:
            pickle.dump(emp_pop_group, f)
        
        # Save adoption curves by class
        emp_curve_group = {}
        emp_curve_group[0] = low_county_curve['cum_reg'].values
        emp_curve_group[1] = mid_county_curve['cum_reg'].values
        emp_curve_group[2] = high_county_curve['cum_reg'].values
        with open(os.path.realpath(os.path.join( self.DATA_PATH,  'emp_curve_group.pkl')), 'wb') as f:
            pickle.dump(emp_curve_group, f)
        
        # Fit Bass diffusion parameters for each class
        start = 0
        t = np.array(range(start,end))
        emp_seed_group = {} 
        
        m = pop_low['POPULATION'].sum()
        if m == 0:
            emp_seed_group[0]= [-1,-1]
        else:
            y_true = low_county_curve['cum_reg'].values[start:end]
            p,q = bassfit(t,y_true,m)
            emp_seed_group[0]= [p,q]
        
        m = pop_mid['POPULATION'].sum()
        if m == 0:
            emp_seed_group[1]= [-1,-1]
        else:
            y_true = mid_county_curve['cum_reg'].values[start:end]
            p,q = bassfit(t,y_true,m)
            emp_seed_group[1]= [p,q]

        m = pop_high['POPULATION'].sum()
        if m == 0:
            emp_seed_group[2]= [-1,-1]
        else:
            y_true = high_county_curve['cum_reg'].values[start:end]
            p,q = bassfit(t,y_true,m)
            emp_seed_group[2] = [p,q]
        
        # Save Bass p,q seeds
        with open(os.path.realpath(os.path.join( self.DATA_PATH, 'emp_seed_group.pkl')), 'wb') as f:
            pickle.dump(emp_seed_group, f)

    # ---------------------------------------------------------------
    # Create benchmark adoption trajectories at tract level
    # using Bass model scaled by tract population ratios.
    # ---------------------------------------------------------------
    def createBenchmark(self):
        tract_curve = pd.read_csv(os.path.join(self.DATA_PATH, 'tract_curve.csv'),converters={'tract': str})       
        demo_tract = pd.read_csv(os.path.join(self.DATA_PATH, 'demo_data.csv'),converters={'GEOID': str})
        
        feature = 'INCOME'; start = 0; end = 13; target = 100
        # by not adding fillna, we keep only few tracts 
        demo_vis = tract_curve.merge(demo_tract,left_on='tract',right_on='GEOID',how='left')#.fillna(0)
        
        ea_low = demo_vis[demo_vis['CLASS'] == 'Low']
        ea_mid = demo_vis[demo_vis['CLASS'] == 'Middle']
        ea_high = demo_vis[demo_vis['CLASS'] == 'High']

        pop_low = demo_tract[demo_tract['CLASS']  == 'Low']
        pop_mid = demo_tract[demo_tract['CLASS'] == 'Middle']
        pop_high = demo_tract[demo_tract['CLASS']  == 'High']
        
        # Compute relative population ratios
        pop_low['pop_ratio'] = pop_low['POPULATION']/(pop_low['POPULATION'].sum()+1e-5)
        pop_mid['pop_ratio'] = pop_mid['POPULATION']/(pop_mid['POPULATION'].sum()+1e-5)
        pop_high['pop_ratio'] = pop_high['POPULATION']/(pop_high['POPULATION'].sum()+1e-5)
        
        # County-level cumulative adoption curves
        low_county_curve = pd.pivot_table(ea_low, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        mid_county_curve = pd.pivot_table(ea_mid, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        high_county_curve = pd.pivot_table(ea_high, values=['cum_reg','dif_reg'], index=['year'], aggfunc=np.sum).reset_index()
        
        if len(low_county_curve)==0:
            low_county_curve = pd.DataFrame({'year':list(range(end)),'cum_reg':np.zeros(end),'dif_reg':np.zeros(end)})
        if len(mid_county_curve)==0:
            mid_county_curve = pd.DataFrame({'year':list(range(end)),'cum_reg':np.zeros(end),'dif_reg':np.zeros(end)})
        if len(high_county_curve)==0:
            high_county_curve = pd.DataFrame({'year':list(range(end)),'cum_reg':np.zeros(end),'dif_reg':np.zeros(end)})

        # Fit Bass model for each class and simulate to target horizon
        t = np.array(range(start,end))
        m = pop_low['POPULATION'].sum()
        y_true = low_county_curve['cum_reg'].values[start:end]
        p,q = bassfit(t,y_true,m)
        A = 1/p
        t = np.array(range(start,target))
        bass_low = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))

        t = np.array(range(start,end))
        m = pop_mid['POPULATION'].sum()
        y_true = mid_county_curve['cum_reg'].values[start:end]
        p,q = bassfit(t,y_true,m)
        A = 1/p
        t = np.array(range(start,target))
        bass_mid = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))

        t = np.array(range(start,end))
        m = pop_high['POPULATION'].sum()
        y_true = high_county_curve['cum_reg'].values[start:end]
        p,q = bassfit(t,y_true,m)
        A = 1/p
        t = np.array(range(start,target))
        bass_high = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))  

        # Scale Bass curves to tracts proportionally by population ratio
        pop_low_year = pop_low.copy()
        pop_mid_year = pop_mid.copy()
        pop_high_year = pop_high.copy()
        for i in range(target):
            pop_low_year[i] = bass_low[i]*pop_low['pop_ratio']
            pop_mid_year[i] = bass_mid[i]*pop_mid['pop_ratio']
            pop_high_year[i] = bass_high[i]*pop_high['pop_ratio']
            
        benchmark_tract_year = pd.concat([pop_low_year,pop_mid_year,pop_high_year])
        benchmark_tract_year = benchmark_tract_year[['GEOID']+list(range(target))]
        benchmark_tract_year.to_csv(os.path.join( self.DATA_PATH, 'benchmark_tract.csv'),index=False)

    # ---------------------------------------------------------------
    # Create state-level benchmark (aggregated across counties)
    # - Builds state-wide demo_data.csv if absent using ACS/vehicle data
    # - Computes Bass p,q for low/mid/high across the state
    # - Allocates state-level Bass forecast back to all tracts
    # - Saves state-wide benchmark and a county-specific slice
    # ---------------------------------------------------------------
    def createBenchmarkState(self):
        emp = self.emp.copy()
        emp['date'] = pd.to_datetime(emp['date'])
        
        if os.path.isfile(
            os.path.join('..','data',self.state_file,'benchmark_state_tract.csv')
        ):  
            # If state benchmark already exists, reuse it
            benchmark_tract_year = pd.read_csv(os.path.join('..','data',self.state_file,'benchmark_state_tract.csv'))
        else:
            # Build state-wide demographic file from ACS income & vehicle tables
            county_fips = pd.read_csv(os.path.join('..','data',self.state_file, 'fips-by-state.csv'),encoding= 'unicode_escape')
            if self.state_file == 'ca':
                income_tract = pd.read_csv(os.path.join('..','data',self.state_file, 'income/ACSST5Y2019.S1901-Data.csv'),skiprows=1)
                vehicle_tract = pd.read_csv(os.path.join('..','data',self.state_file, 'vehicle/ACSDT5Y2019.B08201-Data.csv'),skiprows=1)
            else:
                income_tract = pd.read_csv(os.path.join('..','data',self.state_file, 'income/ACSST5Y2020.S1901-Data.csv'),skiprows=1)
                vehicle_tract = pd.read_csv(os.path.join('..','data',self.state_file, 'vehicle/ACSDT5Y2020.B08201-Data.csv'),skiprows=1)
            demo_name = ['GEOID','INCOME','POPULATION','COUNTY']
            
            demo_tract = income_tract.merge(vehicle_tract,right_on='Geography',left_on='Geography')
            demo_tract['GEOID'] = demo_tract['Geography'].str.split('US',expand=True)[1]
            demo_tract['INCOME'] = demo_tract['Estimate!!Households!!Median income (dollars)']
            # Estimate persons by vehicle availability (1*1 + 2*2 + 3*3 + 4*4 or more)
            demo_tract['POPULATION'] = demo_tract['Estimate!!Total:!!1 vehicle available']+demo_tract['Estimate!!Total:!!2 vehicles available']*2+demo_tract['Estimate!!Total:!!3 vehicles available']*3+demo_tract['Estimate!!Total:!!4 or more vehicles available']*4
            demo_tract = demo_tract[['GEOID','INCOME','POPULATION']]

            county_fips = county_fips[county_fips['state']==self.state_name]
            county_fips['fips'] = county_fips.fips.astype('str')
            if len(county_fips['fips'].values[0]) < 5:
                county_fips['fips'] = '0'+county_fips['fips']

            county_fips = county_fips[county_fips['state']==self.state_name]
            county_fips['fips'] = county_fips.fips.astype('str')
            if len(county_fips['fips'].values[0]) < 5:
                county_fips['fips'] = '0'+county_fips['fips']

            # Map tracts to counties via FIPS
            demo_tract['FIPS'] = demo_tract['GEOID'].str[:5]
            county_tract = demo_tract.merge(county_fips,right_on='fips',left_on='FIPS')[['name','GEOID']]
            county_tract['COUNTY'] = county_tract.name.str.split(' County',expand=True)[0]
            county_tract['GEOID'] = county_tract['GEOID'].astype(str)
            county_tract = county_tract[['COUNTY','GEOID']]
            county_tract.to_csv(os.path.join(self.DATA_PATH, 'county_tract.csv'),index=False)
            demo_tract = demo_tract.merge(county_tract,left_on='GEOID',right_on='GEOID')
            
            # Clean and interpolate INCOME; keep POPULATION numeric
            demo_tract.loc[demo_tract['INCOME']=='-', 'INCOME'] = np.nan
            demo_tract.loc[demo_tract['POPULATION']==0, 'INCOME'] = 0
            demo_tract.loc[demo_tract['INCOME']=='250,000+', 'INCOME'] = 250000
            demo_tract['INCOME'] = demo_tract['INCOME'].astype(float)
            demo_tract['INCOME'] = demo_tract['INCOME'].interpolate(method='polynomial', order=2)
            demo_tract['GEOID'] = demo_tract['GEOID'].astype(str)
            demo_tract['POPULATION'] = demo_tract['POPULATION'].astype(float)
            
            # State-specific income thresholds for class assignment
            if self.state_file == 'ca':
                thr1 = 64623; thr2 = 95587
            if self.state_file == 'wa':
                thr1 = 66685; thr2 = 92830
            
            start = 0; end = 13; target = 50; feature = 'INCOME'
            demo_tract.loc[demo_tract[feature] <= thr1, 'CLASS'] = 'Low'
            demo_tract.loc[(demo_tract[feature] > thr1)&(demo_tract[feature] <= thr2), 'CLASS'] = 'Middle'
            demo_tract.loc[demo_tract[feature] > thr2, 'CLASS'] = 'High'
            
            # Keep only tracts present in shapefile; save state-wide demo
            demo_tract = demo_tract.merge(self.tract_shp,right_on='GEOID',left_on='GEOID')[['GEOID','CLASS','POPULATION','COUNTY','INCOME']]
            demo_tract.to_csv(os.path.join('..','data',self.state_file,'demo_data.csv'),index=False)
        
            # Build tract-level cumulative curves across entire state
            key ='tract'
            emp_year = []
            value_list = emp[key].dropna().unique()
            for value in value_list:
                emp_grouped_year = year_groupby(emp,key,value)
                emp_grouped_year = emp_grouped_year[[key,'year','dif_reg','cum_reg']]
                emp_year.append(emp_grouped_year)
            emp_year_key = pd.concat(emp_year)
            emp_year_key = emp_year_key[emp_year_key['tract'].isin(demo_tract.GEOID.values)]
            tract_curve = emp_year_key.copy()
            
            # Thresholds again for convenience
            if self.state_file == 'ca':
                thr1 = 64623; thr2 = 95587
            if self.state_file == 'wa':
                thr1 = 66685; thr2 = 92830
            
            feature = 'INCOME'; start = 0; end = 13; target = 100
            demo_vis = tract_curve.merge(demo_tract,left_on='tract',right_on='GEOID',how='left')
            
            # Split by income thresholds (not by CLASS here)
            ea_low = demo_vis[demo_vis[feature] <= thr1]
            ea_mid = demo_vis[(demo_vis[feature] > thr1)&(demo_vis[feature] <= thr2)]
            ea_high = demo_vis[demo_vis[feature] > thr2]

            pop_low = demo_tract[demo_tract[feature] <= thr1]
            pop_mid = demo_tract[(demo_tract[feature] > thr1)&(demo_tract[feature] <= thr2)]
            pop_high = demo_tract[demo_tract[feature] > thr2]
            
            # Population normalization within each state-wide class
            pop_low['pop_ratio'] = pop_low['POPULATION']/pop_low['POPULATION'].sum()
            pop_mid['pop_ratio'] = pop_mid['POPULATION']/pop_mid['POPULATION'].sum()
            pop_high['pop_ratio'] = pop_high['POPULATION']/pop_high['POPULATION'].sum()
            
            # State-wide cumulative adoption curves per income slice
            low_county_curve = pd.pivot_table(ea_low, values=['cum_reg'], index=['year'], aggfunc=np.sum).reset_index()
            mid_county_curve = pd.pivot_table(ea_mid, values=['cum_reg'], index=['year'], aggfunc=np.sum).reset_index()
            high_county_curve = pd.pivot_table(ea_high, values=['cum_reg'], index=['year'], aggfunc=np.sum).reset_index()
            
            # Fit Bass and simulate to target for each slice
            pq_value = {}
            t = np.array(range(start,end))
            m = pop_low['POPULATION'].sum()
            y_true = low_county_curve['cum_reg'].values[start:end]
            p,q = bassfit(t,y_true,m)
            A = 1/p
            t = np.array(range(start,target))
            bass_low = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))
            pq_value['low']={}
            pq_value['low']['p'] = p
            pq_value['low']['q'] = q
            pq_value['low']['pq'] = p/q

            t = np.array(range(start,end))
            m = pop_mid['POPULATION'].sum()
            y_true = mid_county_curve['cum_reg'].values[start:end]
            p,q = bassfit(t,y_true,m)
            A = 1/p
            t = np.array(range(start,target))
            bass_mid = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))
            pq_value['middle']={}
            pq_value['middle']['p'] = p
            pq_value['middle']['q'] = q
            pq_value['middle']['pq'] = p/q

            t = np.array(range(start,end))
            m = pop_high['POPULATION'].sum()
            y_true = high_county_curve['cum_reg'].values[start:end]
            p,q = bassfit(t,y_true,m)
            A = 1/p
            t = np.array(range(start,target))
            bass_high = m*(1-A*(np.exp(-(p+q)*t)*p+np.exp(-(p+q)*t)*q)/(1+A*q*np.exp(-(p+q)*t)))  
            pq_value['high']={}
            pq_value['high']['p'] = p
            pq_value['high']['q'] = q
            pq_value['high']['pq'] = p/q

            # Save state-wide p,q values per income slice
            with open(os.path.realpath(os.path.join( '..','data', self.state_file, 'emp_pq_group.pkl')), 'wb') as f:
                pickle.dump(pq_value, f)

            # Allocate state-wide Bass curves back to all tracts by pop ratio
            pop_low_year = pop_low.copy()
            pop_mid_year = pop_mid.copy()
            pop_high_year = pop_high.copy()
            for i in range(target):
                pop_low_year[i] = bass_low[i]*pop_low['pop_ratio']
                pop_mid_year[i] = bass_mid[i]*pop_mid['pop_ratio']
                pop_high_year[i] = bass_high[i]*pop_high['pop_ratio']
                
            benchmark_tract_year = pd.concat([pop_low_year,pop_mid_year,pop_high_year])
            benchmark_tract_year = benchmark_tract_year[['GEOID','COUNTY']+list(range(target))]
            benchmark_tract_year.to_csv(os.path.join('..','data',self.state_file,'benchmark_state_tract.csv'),index=False)
        
        # Save county-specific slice of the state-wide benchmark
        benchmark_tract_year = benchmark_tract_year[benchmark_tract_year['COUNTY']==self.county_name]
        benchmark_tract_year.to_csv(os.path.join('..','data',self.state_file,self.county_name,'benchmark_state_tract.csv'),index=False)
        
    # ---------------------------------------------------------------
    # Create tract-to-tract distance matrix M.npy
    # - Diagonal: effective "radius" = sqrt(area)/2 (km) as a proxy
    # - Off-diagonal: great-circle distance between tract centroids
    # ---------------------------------------------------------------
    def createM(self):
        tract_coord = pd.read_csv(os.path.join(self.DATA_PATH, 'tract_coord.csv'))
        tract_coord = tract_coord.reset_index()
        n_tract = tract_coord['GEOID'].nunique()

        M = np.zeros((n_tract, n_tract))
        for idx_start in tqdm(np.arange(M.shape[0])):
            for idx_end in np.arange(M.shape[1]):
                if idx_start == idx_end:
                    M[idx_start][idx_end] = np.sqrt(tract_coord[tract_coord['index'] == idx_start]['Area'].iloc[0])/2
                else:
                    start = (
                        tract_coord[tract_coord['index'] == idx_start]['Latitude'].iloc[0], 
                        tract_coord[tract_coord['index'] == idx_start]['Longitude'].iloc[0]
                    )

                    end = (
                        tract_coord[tract_coord['index'] == idx_end]['Latitude'].iloc[0], 
                        tract_coord[tract_coord['index'] == idx_end]['Longitude'].iloc[0]
                    )

                    M[idx_start][idx_end] = great_circle(start, end).kilometers

        np.save(os.path.join(self.DATA_PATH, 'M.npy'), M)
        
    # ---------------------------------------------------------------
    # Orchestrate the full data-prep pipeline for a county:
    #   1) createBenchmarkState (state-wide baseline + county slice)
    #   2) createDemo           (county demo & coordinates)
    #   3) createM              (distance matrix for county tracts)
    #   4) createCurve          (empirical curves, Bass seeds)
    #   5) createBenchmark      (county Bass benchmark per tract)
    # ---------------------------------------------------------------
    def createData(self):
        self.createBenchmarkState()
        self.createDemo()
        self.createM()
        self.createCurve()
        self.createBenchmark()