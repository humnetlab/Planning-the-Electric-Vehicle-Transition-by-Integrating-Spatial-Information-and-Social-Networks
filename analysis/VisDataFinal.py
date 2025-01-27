import numpy as np
import pandas as pd
import scipy as sp
import os
import pickle
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import geopandas as gpd
import matplotlib as mpl
import rioxarray
import xarray
from pysal.explore import esda
from pysal.lib import weights
from esda.moran import Moran
import contextily
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
from libpysal.weights.contiguity import Queen
from libpysal import examples
from splot.esda import moran_scatterplot
from esda.moran import Moran_Local
from splot.esda import plot_local_autocorrelation
from splot._viz_utils import mask_local_auto, moran_hot_cold_spots, splot_colors
from matplotlib import colors, patches
import matplotlib.colors as clr
import contextily as cx
from itertools import cycle
warnings.filterwarnings('ignore')
plt.style.use("seaborn-white")
sns.set_style('ticks')

color_platte = ['#00429d', '#76a6c8', '#bee8de', '#fbbd79', '#d27144', '#93003a']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_platte)

colors= ['#001144', '#00245a', '#013871', '#0d4c89', '#305f9e', '#4973b4', '#6087ca', '#779ce1', '#8eb2f9', '#a7caff', '#c1e4ff'][::-1]
nodes = np.arange(0,1.1,0.1)
bcmap = clr.LinearSegmentedColormap.from_list("bcmap", list(zip(nodes, colors)))

colors= ['#001819', '#012b2e', '#0a3e41', '#235255', '#386669', '#4d7b7e', '#639194', '#78a7aa', '#8fbec1', '#a5d5d8', '#bdedf0'][::-1]
nodes = np.arange(0,1.1,0.1)
gcmap = clr.LinearSegmentedColormap.from_list("gcmap", list(zip(nodes, colors)))

colors= ['#ff0000', '#ff4713', '#ff662a', '#ff7f40', '#ff9456', '#ffa86d', '#ffba83', '#ffcc9a', '#ffdeb1', '#ffeec8', '#ffffe0'][::-1]
nodes = np.arange(0,1.1,0.1)
ycmap = clr.LinearSegmentedColormap.from_list("ycmap", list(zip(nodes, colors)))

colors= ['#440000', '#580000', '#700401', '#892216', '#a23929', '#bc4f3c', '#d66550', '#f17c65', '#ff937b', '#ffab91', '#ffc3a8'][::-1]
nodes = np.arange(0,1.1,0.1)
rcmap = clr.LinearSegmentedColormap.from_list("rcmap", list(zip(nodes, colors)))

colors= ['#00429d', '#73a2c6', '#ffffe0', '#e0884e', '#93003a']
nodes = np.arange(0,1.1,0.25)
catcmap = clr.LinearSegmentedColormap.from_list("catcmap", list(zip(nodes, colors)))

import matplotlib.colors as clr
colors= ['#00429d', '#2451a4', '#3761ab', '#4771b2', '#5681b9', '#6492c0', '#73a2c6', '#82b3cd', '#93c4d2', '#a5d5d8', '#b9e5dd', '#d3f4e0', '#ffffe0', '#ffebba', '#ffd699', '#fcc17e', '#f4ae69', '#eb9b5a', '#e0884e', '#d57545', '#c9623f', '#bc4f3c', '#af3b3a', '#a1253a', '#93003a']
nodes = np.arange(0,1.04,0.04)
heatcmap = clr.LinearSegmentedColormap.from_list("heatcmap", list(zip(nodes, colors)))

def lisa_cluster(
    city, moran_loc, gdf, p=0.05, ax=None, legend=True, legend_kwds=None, **kwargs
):
    _, colors5, _, labels = mask_local_auto(moran_loc, p=p)

    hmap = clr.ListedColormap(colors5)

    if ax is None:
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        gdf = gdf.assign(cl=labels)
        gdf = gpd.clip(gdf, city)
        
        gdf.plot(
            column="cl",
            categorical=True,
            k=2,
            cmap=hmap,
            linewidth=0.1,
            ax=ax,
            edgecolor="white",
            legend=legend,
            legend_kwds=legend_kwds,
            **kwargs
        )
    else:
        gdf = gdf.assign(cl=labels)
        gdf = gpd.clip(gdf, city)
        gdf.plot(
            column="cl",
            categorical=True,
            k=2,
            cmap=hmap,
            linewidth=1.5,
            ax=ax,
            legend=legend,
            legend_kwds=legend_kwds,
            **kwargs
        )

    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


class VisData:
    def __init__(self, state, counties):
        self.state = state
        self.counties = counties
        self.RESULT_PATH = os.path.join('..', 'result', self.state)
        self.FIGURE_PATH = os.path.join('..', 'figure', self.state)
        self.county_shp = gpd.read_file(os.path.join('..','data',self.state,'county','tl_2019_us_county.shp'))
        if self.state == 'ca':
            self.tract_shp = gpd.read_file(os.path.join('..','data',self.state,'tract','tl_2019_06_tract.shp'))
            self.county_shp = self.county_shp[self.county_shp['STATEFP']=='06']
            self.place_shp = gpd.read_file(os.path.join('..','data',self.state,'place','tl_rd22_06_place.shp'))
        if self.state == 'wa':
            self.tract_shp = gpd.read_file(os.path.join('..','data',self.state,'tract','tl_2020_53_tract.shp'))
            self.county_shp = self.county_shp[self.county_shp['STATEFP']=='53']
            self.place_shp = gpd.read_file(os.path.join('..','data',self.state,'place','tl_2020_53_place20.shp'))

    def loadData(self):
        data= []; abm_names = []; de_names = []; emp_names = []; ds_names = []
        end = 100; batch_num=3
        for i in range(end):
            abm_names.append('ABM_'+str(i))
        for i in range(end):
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        
        de_curve_state = pd.read_csv(os.path.join( '..', 'data', self.state, 'benchmark_state_tract.csv'),converters={'GEOID': str})
        for county in self.counties:
            demo = pd.read_csv(os.path.join( '..', 'data', self.state, county, 'demo_data.csv'),converters={'GEOID': str})
            emp_curve = pd.read_csv(os.path.join( '..', 'data', self.state, county, 'tract_curve.csv'),converters={'tract': str})
            de_curve = pd.read_csv(os.path.join( '..', 'data', self.state, county, 'benchmark_tract.csv'),converters={'GEOID': str})
            ds_curve = de_curve_state[de_curve_state['COUNTY']==county].copy().drop(['COUNTY'], axis=1)
            
            emp_curve = pd.pivot_table(emp_curve, values='cum_reg', index=['tract'], columns=['year'], aggfunc="sum", fill_value=0).reset_index()
            emp_curve.columns = ['GEOID']+emp_names
            de_curve.columns = ['GEOID']+de_names
            ds_curve.columns = ['GEOID']+ds_names
            
            pop = pd.read_csv(os.path.realpath(os.path.join('..', 'data', self.state, county, 'demo_data.csv')),converters={'GEOID': str}).POPULATION.sum()
            number_node = int(pop/5)
            emp_pop_group = pickle.load(open(os.path.realpath(os.path.join( '..', 'data', self.state, county, 'emp_pop_group.pkl')), 'rb'), encoding='bytes')
            scale = number_node/pop

            low_curve = []; mid_curve = []; high_curve = []
            for j in range(batch_num):
                FOLDER_PATH = 'fit_pq_'+'state_'+''.join(i.lower() for i in county.split())+'_'+str(j)+'_'+str(number_node)+'_'+str(-1)+'_'+str(10)+'_'+str(3)+'_'+str(3)
                for i in range(10):
                    sub_tract = (pd.read_csv(os.path.join(self.RESULT_PATH, county, FOLDER_PATH, 'curves', 'tract_curve_'+str(i)+'.csv'),index_col=0).cumsum(axis=0).transpose()/scale)
                    low = emp_pop_group[0].merge(sub_tract,left_on='GEOID',right_on=sub_tract.index)[['GEOID']+list(range(end))]
                    low['REP'] = i; low['BATCH'] = j; low_curve.append(low)
                    mid = emp_pop_group[1].merge(sub_tract,left_on='GEOID',right_on=sub_tract.index)[['GEOID']+list(range(end))]
                    mid['REP'] = i; mid['BATCH'] = j; mid_curve.append(mid)
                    high = emp_pop_group[2].merge(sub_tract,left_on='GEOID',right_on=sub_tract.index)[['GEOID']+list(range(end))]
                    high['REP'] = i; high['BATCH'] = j; high_curve.append(high)
            
            abm_curve = pd.concat(low_curve+mid_curve+high_curve)
            abm_curve.columns = ['GEOID']+abm_names+['REP','BATCH']
            data.append(demo.merge(ds_curve, right_on='GEOID', left_on='GEOID',how='outer').merge(de_curve, right_on='GEOID', left_on='GEOID',how='outer').merge(emp_curve,left_on='GEOID',right_on='GEOID',how='outer').merge(abm_curve,left_on='GEOID',right_on='GEOID',how='outer'))
        
        data = pd.concat(data)
        data = data[['GEOID','COUNTY','POPULATION','CLASS','REP','BATCH','INCOME']+emp_names+abm_names+de_names+ds_names]
        data.to_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),index=False)
        
        p_low_de = []; p_middle_de = []; p_high_de = []
        q_low_de = []; q_middle_de = []; q_high_de = []
        p_low_abm = []; p_middle_abm = []; p_high_abm = []
        q_low_abm = []; q_middle_abm = []; q_high_abm = []
        county_pq = []
        
        for county in self.counties:
            pop = pd.read_csv(os.path.realpath(os.path.join('..', 'data', self.state, county, 'demo_data.csv')),converters={'GEOID': str}).POPULATION.sum()
            emp_pop_group = pickle.load(open(os.path.realpath(os.path.join( '..', 'data', self.state, county, 'emp_seed_group.pkl')), 'rb'), encoding='bytes')
            number_node = int(pop/5)
            for j in range(batch_num):
                folder = 'fit_pq_'+'state_'+''.join(i.lower() for i in county.split())+'_'+str(j)+'_'+str(number_node)+'_'+str(-1)+'_'+str(10)+'_'+str(3)+'_'+str(3)
                p_value_star = pickle.load(open(os.path.realpath(os.path.join( '..', 'result', self.state, county, folder, 'parameters','p_value_star.pkl')), 'rb'), encoding='bytes')
                q_value_star = pickle.load(open(os.path.realpath(os.path.join( '..', 'result', self.state, county, folder,'parameters','q_value_star.pkl')), 'rb'), encoding='bytes')
                p_low_de.append(emp_pop_group[0][0]); p_middle_de.append(emp_pop_group[1][0]); p_high_de.append(emp_pop_group[2][0])
                q_low_de.append(emp_pop_group[0][1]); q_middle_de.append(emp_pop_group[1][1]); q_high_de.append(emp_pop_group[2][1])
                p_low_abm.append(p_value_star[0]); p_middle_abm.append(p_value_star[1]); p_high_abm.append(p_value_star[2])
                q_low_abm.append(q_value_star[0]); q_middle_abm.append(q_value_star[1]); q_high_abm.append(q_value_star[2])
                county_pq.append(county)
        pq = pd.DataFrame({'COUNTY':county_pq, 'p_low_de': p_low_de, 'p_middle_de':p_middle_de, 'p_high_de':p_high_de, 'q_low_de':q_low_de, 'q_middle_de':q_middle_de, 'q_high_de':q_high_de, 'p_low_abm':p_low_abm, 'p_middle_abm':p_middle_abm, 'p_high_abm':p_high_abm, 'q_low_abm':q_low_abm, 'q_middle_abm': q_middle_abm, 'q_high_abm':q_high_abm})
        pq.to_csv(os.path.join( '..', 'data', self.state, self.state+'_pq.csv'),index=False)

    def visCounty(self):
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        ## 1 Temporal: Curves of cumulative adoptions (DE/ABM/EMP, 3 groups, 2010-2022)
        data_temp = data.copy()
        abm_names=[]; de_names=[]; emp_names=[]; ds_names=[]
        for i in range(13):
            abm_names.append('ABM_'+str(i))
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        
        low_b = 5; mid_b = 50; high_b = 95
        for county in self.counties:
            data_temp = data.copy()
            data_temp = data_temp[data_temp['COUNTY']==county]
            data_temp_low = data_temp[data_temp['CLASS']=='Low']
            data_temp_mid = data_temp[data_temp['CLASS']=='Middle']
            data_temp_high = data_temp[data_temp['CLASS']=='High']
            
            data_temp_low = pd.pivot_table(data_temp_low, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY','REP','BATCH'], aggfunc="sum").reset_index()
            data_temp_mid = pd.pivot_table(data_temp_mid, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY','REP','BATCH'], aggfunc="sum").reset_index()
            data_temp_high = pd.pivot_table(data_temp_high, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY','REP','BATCH'], aggfunc="sum").reset_index()
                    
            fig = plt.figure(figsize=(10,3))
            axs = fig.add_subplot(1, 3, 1)
            dtl = data_temp_low[data_temp_low['COUNTY']==county]
            
            try:
                pop_low = data_temp_low.drop_duplicates(subset=['COUNTY'])['POPULATION'].sum()
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[de_names].values[0]/pop_low, color = 'g', alpha= 0.5, label = 'DE' )
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[ds_names].values[0]/pop_low, color = 'b', alpha= 0.5, label = 'DS' )
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[emp_names].values[0]/pop_low,  color = 'k', alpha= 0.5, label = 'EMP' )
                p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
                p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
                p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
                plt.plot(np.arange(2010,2023),p_50/pop_low, color = 'r', alpha= 0.5, label = 'ABM')
                plt.fill_between(np.arange(2010,2023), p_5/pop_low, p_95/pop_low, alpha=0.2)

                plt.legend()
                plt.xlabel('Year')
                plt.ylabel('Adoption Rate')
                plt.title(county+',Low')
            except:
                print()

            axs = fig.add_subplot(1, 3, 2)
            dtl = data_temp_mid[data_temp_mid['COUNTY']==county]
            try:
                pop_mid = data_temp_mid.drop_duplicates(subset=['COUNTY'])['POPULATION'].sum()
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[emp_names].values[0]/pop_mid,  color = 'k', alpha= 0.5, label = 'EMP' )
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[de_names].values[0]/pop_mid, color = 'g', alpha= 0.5, label = 'DE' )
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[ds_names].values[0]/pop_mid, color = 'b', alpha= 0.5, label = 'DS' )
                p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
                p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
                p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
                plt.plot(np.arange(2010,2023),p_50/pop_mid, color = 'r', alpha= 0.5, label = 'ABM')
                plt.fill_between(np.arange(2010,2023), p_5/pop_mid, p_95/pop_mid, alpha=0.2)

                plt.legend()
                plt.xlabel('Year')
                plt.ylabel('Adoption Rate')
                plt.title(county+',Middle')
            except:
                print()

            axs = fig.add_subplot(1, 3, 3)
            dtl = data_temp_high[data_temp_high['COUNTY']==county]
            try:
                pop_high = data_temp_high.drop_duplicates(subset=['COUNTY'])['POPULATION'].sum()
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[de_names].values[0]/pop_high, color = 'g', alpha= 0.5, label = 'DE' )
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[ds_names].values[0]/pop_high, color = 'b', alpha= 0.5, label = 'DS' )
                plt.plot(np.arange(2010,2023),dtl.drop_duplicates(subset=['COUNTY'])[emp_names].values[0]/pop_high,  color = 'k', alpha= 0.5, label = 'EMP' )
                p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
                p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
                p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
                plt.plot(np.arange(2010,2023),p_50/pop_high, color = 'r', alpha= 0.5, label = 'ABM')
                plt.fill_between(np.arange(2010,2023), p_5/pop_high, p_95/pop_high, alpha=0.2)

                plt.legend()
                plt.xlabel('Year')
                plt.ylabel('Adoption Rate')
                plt.title(county+',High')
            except:
                print()
            
            plt.tight_layout()
            if not os.path.exists(os.path.join( self.FIGURE_PATH, county)):
                os.makedirs(os.path.join( self.FIGURE_PATH, county))
            plt.savefig(os.path.join( self.FIGURE_PATH, county, 'now_curve_county.png'))
            plt.show()
            
        ## 2 Temporal: Curves of cumulative adoptions (DE/ABM/EMP, 3 groups, 2010-2022)
        data_temp = data.copy()
        abm_names=[]; de_names=[]; emp_names=[]; ds_names=[]
        end = 40
        for i in range(end):
            abm_names.append('ABM_'+str(i))
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))
        
        low_b = 5; mid_b = 50; high_b = 95
        for county in self.counties:
            data_temp = data.copy()
            data_temp_low = data_temp[data_temp['CLASS']=='Low']
            data_temp_mid = data_temp[data_temp['CLASS']=='Middle']
            data_temp_high = data_temp[data_temp['CLASS']=='High']
            
            data_temp_low = pd.pivot_table(data_temp_low, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY','REP'], aggfunc="sum", fill_value=0).reset_index()
            data_temp_mid = pd.pivot_table(data_temp_mid, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY','REP'], aggfunc="sum", fill_value=0).reset_index()
            data_temp_high = pd.pivot_table(data_temp_high, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY','REP'], aggfunc="sum", fill_value=0).reset_index()         
            
            fig = plt.figure(figsize=(10,3)) 
            axs = fig.add_subplot(1, 3, 1)
            dtl = data_temp_low[data_temp_low['COUNTY']==county]
            
            try:
                pop_low = data_temp_low.drop_duplicates(subset=['COUNTY'])['POPULATION'].sum()
                plt.plot(np.arange(2010,2010+end),dtl.drop_duplicates(subset=['COUNTY'])[de_names].values[0]/pop_low, color = 'g', alpha= 0.5, label = 'DE' )
                plt.plot(np.arange(2010,2010+end),dtl.drop_duplicates(subset=['COUNTY'])[ds_names].values[0]/pop_low, color = 'b', alpha= 0.5, label = 'DS' )
                p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
                p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
                p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
                plt.plot(np.arange(2010,2010+end),p_50/pop_low, color = 'r', alpha= 0.5, label = 'ABM')
                plt.fill_between(np.arange(2010,2010+end), p_5/pop_low, p_95/pop_low, alpha=0.2)

                plt.legend()
                plt.xlabel('Year')
                plt.ylabel('Adoption Rate')
                plt.title(county+',Low')
            except:
                print()

            axs = fig.add_subplot(1, 3, 2)
            dtl = data_temp_mid[data_temp_mid['COUNTY']==county]
            try:
                pop_mid = data_temp_mid.drop_duplicates(subset=['COUNTY'])['POPULATION'].sum()
                plt.plot(np.arange(2010,2010+end),dtl.drop_duplicates(subset=['COUNTY'])[de_names].values[0]/pop_mid, color = 'g', alpha= 0.5, label = 'DE' )
                plt.plot(np.arange(2010,2010+end),dtl.drop_duplicates(subset=['COUNTY'])[ds_names].values[0]/pop_mid, color = 'b', alpha= 0.5, label = 'DS' )
                p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
                p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
                p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
                plt.plot(np.arange(2010,2010+end),p_50/pop_mid, color = 'r', alpha= 0.5, label = 'ABM')
                plt.fill_between(np.arange(2010,2010+end), p_5/pop_mid, p_95/pop_mid, alpha=0.2)

                plt.legend()
                plt.xlabel('Year')
                plt.ylabel('Adoption Rate')
                plt.title(county+',Middle')
            except:
                print()

            axs = fig.add_subplot(1, 3, 3)
            dtl = data_temp_high[data_temp_high['COUNTY']==county]
            try:
                pop_high = data_temp_high.drop_duplicates(subset=['COUNTY'])['POPULATION'].sum()
                plt.plot(np.arange(2010,2010+end),dtl.drop_duplicates(subset=['COUNTY'])[de_names].values[0]/pop_high, color = 'g', alpha= 0.5, label = 'DE' )
                plt.plot(np.arange(2010,2010+end),dtl.drop_duplicates(subset=['COUNTY'])[ds_names].values[0]/pop_high, color = 'b', alpha= 0.5, label = 'DS' )
                p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
                p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
                p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
                plt.plot(np.arange(2010,2010+end),p_50/pop_high, color = 'r', alpha= 0.5, label = 'ABM')
                plt.fill_between(np.arange(2010,2010+end), p_5/pop_high, p_95/pop_high, alpha=0.2)

                plt.legend()
                plt.xlabel('Year')
                plt.ylabel('Adoption Rate')
                plt.title(county+',High')
            except:
                print()
            
            plt.tight_layout()
            if not os.path.exists(os.path.join( self.FIGURE_PATH, county)):
                os.makedirs(os.path.join( self.FIGURE_PATH, county))
            plt.savefig(os.path.join( self.FIGURE_PATH, county, 'future_curve_county.png'))
            plt.show()
        
    def currentAdoption(self):
        if self.state == 'ca':
            cs = 'orange'; cm = 'YlOrRd'; cs2 = '#ff0000'
        else:
            cs = 'blue'; cm = bcmap; cs2 = '#81dbed'
            
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        emp_names = []
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        data = data.drop_duplicates(subset=['GEOID','CLASS'])
        data = pd.pivot_table(data, values=emp_names, index=['GEOID'], aggfunc="sum", fill_value=0).reset_index()
        demo = pd.read_csv(os.path.join( '..', 'data', self.state, 'demo_data.csv'),converters={'GEOID': str})
        data = data.merge(demo,left_on='GEOID',right_on='GEOID')
        
        ## 1.1 Curves of cumulative adoptions and new adoptions.
        data_curve = data.copy()
        cdf_curve = (data_curve[emp_names].sum(axis=0)/demo['POPULATION'].sum()).values
        pdf_curve = np.concatenate(([cdf_curve[0]],cdf_curve[1:]-cdf_curve[:-1]),axis=None) 
        
        fig = plt.figure(figsize=(2.7,2)) 
        axs1 = fig.add_subplot()
        
        axs1.plot(range(2010,2023),pdf_curve*100, c=cs2,label='New',marker='d', alpha = 1, linewidth = 0.5)
        axs1.set_ylim(0,0.5)
        axs1.set_ylabel('New Adoption Rate [%]')
        
        axs = axs1.twinx() 
        axs.plot(range(2010,2023,2),cdf_curve[::2]*100, c=cs,label='Total',marker='s')
        axs.set_ylim(0,2.5)
        axs.set_ylabel('Total Adoption Rate [%]')
        axs.set_xlabel('Year')
        
        fig.legend(loc='upper left',bbox_to_anchor=(0.25, 0, 0, 0.9))
        plt.tight_layout()
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_curve_rate.pdf'),dpi=300,bbox_inches='tight')
        print('cdf_curve')
        print(cdf_curve*100)
        print('pdf_curve')
        print(pdf_curve*100)

        ## 1.2 Map of adopters distribution.
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        emp_names = []
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        data = data.drop_duplicates(subset=['GEOID','CLASS'])
        data = pd.pivot_table(data, values=emp_names+['POPULATION'], index=['COUNTY'], aggfunc="sum", fill_value=0).reset_index()

        data_map = data.copy()
        data_map['RATE'] = data_map['EMP_12']/data_map['POPULATION']*100
        data_map = self.county_shp.merge(data_map, left_on = 'NAME', right_on = 'COUNTY')
        fig = plt.figure(figsize=(3.5,3.5)) 
        axs = fig.add_subplot(1, 1, 1)
        ax = data_map.plot(ax = axs, column="RATE", cmap=cm, edgecolor=None, legend=False,linewidth=1)
        if self.state == 'ca':
            ax.set_xlim(-130, -110)
            ax.set_ylim(31, 43)
        else:
            ax.set_xlim(-125, -116.5)
            ax.set_ylim(45, 50)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Adoption Rate [%]')
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=data_map.crs,attribution_size=6)
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_map_rate.pdf'),dpi=300,bbox_inches='tight')
        
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        emp_names = []
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        data = data.drop_duplicates(subset=['GEOID','CLASS'])
        data = pd.pivot_table(data, values=emp_names+['POPULATION'], index=['COUNTY'], aggfunc="sum", fill_value=0).reset_index()

        data_map = data.copy()
        data_map['RATE'] = data_map['EMP_12']/data_map['POPULATION']*100
        data_map = self.county_shp.merge(data_map, left_on = 'NAME', right_on = 'COUNTY')
        fig = plt.figure(figsize=(3,3)) 
        axs = fig.add_subplot(1, 1, 1)
        ax = data_map.plot(ax = axs, column="RATE", cmap=cm, edgecolor=None, legend=False,linewidth=1)
        ax.set_xlim(-122.9, -120)
        ax.set_ylim(36.3, 39)
        ax.set_xticks([])
        ax.set_yticks([])
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_map_rate_zoom_bay.pdf'),dpi=300,bbox_inches='tight')
        
        
        # 1.3 Correlation map
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        emp_names = []
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        data = data.drop_duplicates(subset=['GEOID','CLASS'])
        data = pd.pivot_table(data, values=emp_names+['POPULATION','INCOME'], index=['GEOID'], aggfunc="sum", fill_value=0).reset_index()
        
        data_map = data.copy()
        data_map['RATE'] = data_map['EMP_12']/data_map['POPULATION']*100
        data_map['INCOME'] = data_map['INCOME']/1000
        data_map = self.tract_shp.merge(data_map, left_on = 'GEOID', right_on = 'GEOID')
        data_map = data_map[data_map['RATE']<=15]
        
        g = sns.jointplot(x="INCOME", y="RATE", data=data_map,
                  kind="reg", truncate=True,
                  xlim=(0, 250), ylim = (0,15),
                  color=cs, height=2.2, scatter_kws={'s': 0.05},
                  line_kws={'color': 'black', 'linewidth': 2.5})
        g.set_axis_labels('Income [1,000 $]', 'Adoption Rate [%]')
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_scatter_income.pdf'),dpi=300,bbox_inches='tight')
        
        reg = LinearRegression(fit_intercept=True).fit(data_map['INCOME'].values.reshape(-1, 1), data_map['RATE'].values.reshape(-1, 1))
        print('fit score:', reg.score(data_map['INCOME'].values.reshape(-1, 1), data_map['RATE'].values.reshape(-1, 1)))
        print('fit pearson:', sp.stats.pearsonr(data_map['INCOME'].values, data_map['RATE'].values))
        print('fit parameter:', reg.coef_, reg.intercept_)
    
    def analyzeAdoption(self):
        if self.state=='wa':
            colorv = 'blue'; colorp = 'blue'
            colors = ['#81dbed','#60a3f2','blue','#81dbed','#60a3f2','blue'] 
            county = 'King'
        else:
            colorv = 'orange'; colorp = 'brown'
            colors = ['#ffc34b', '#ff6e00', '#ff0000','#ffc34b', '#ff6e00', '#ff0000']
            county = 'Los Angeles'
        
        # 2.1 boxplot for pq
        pq = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_pq.csv'))
        emp_pq_group = pickle.load(open(os.path.realpath(os.path.join( '..', 'data', self.state, 'emp_pq_group.pkl')), 'rb'), encoding='bytes')
        
        value_pq = pq.replace(-1,np.nan)        
        
        count = 1
        fig = plt.figure(figsize=(7.5,2.7))
        for par in ['p','q']:
            axs = fig.add_subplot(1, 2, count)
            axs.set_xscale("log")
            print('County BM + SocNet BM')
            print(value_pq[[par+'_low_de',par+'_middle_de',par+'_high_de',par+'_low_abm',par+'_middle_abm',par+'_high_abm']].median())
            
            data_pq = pd.melt(pq, id_vars=['COUNTY'], value_vars=[par+'_low_de',par+'_middle_de',par+'_high_de',par+'_low_abm',par+'_middle_abm',par+'_high_abm'], var_name='Model', value_name=par)
            
            data_pq = data_pq.replace(-1,np.nan)            
            gfg = sns.boxplot( data = data_pq, x=par, y="Model", fill = True, linecolor='k', width=0.6, showfliers=True, flierprops={"marker": "+"}, ax =axs)
            gfg.set_yticklabels(['County-Low','County-Middle','County-High','SocNet-Low','SocNet-Middle','SocNet-High'])
            
            hatches = ['*','*','*','o','o','o']
            patches = [patch for patch in gfg.patches if type(patch) == mpl.patches.PathPatch]
            h =  hatches * (len(patches) // len(hatches))
            for patch, color, hatch in zip(patches,colors, h):
                patch.set_color(color)
                patch.set_edgecolor('k')
            
            for name in ['low','middle','high']:
                emp_pq = emp_pq_group[name][par]
                print('State BM,'+par+','+name)
                print(emp_pq)
                if name == 'low':
                    clr = colors[0]
                    axs.axvline(emp_pq, dashes=(2, 2), c = clr, linewidth=2, marker = 'x', label='State-Low')
                if name == 'middle':
                    clr= colors[1]
                    axs.axvline(emp_pq, dashes=(2, 2), c = clr, linewidth=2, marker = 'x', label='State-Middle')
                if name == 'high':
                    clr = colors[2]
                    axs.axvline(emp_pq, dashes=(2, 2), c = clr, linewidth=2, marker = 'x', label='State-High')
                if par == 'p':
                    plt.xlim([6e-8,1e-2])
                if par == 'q':
                    plt.xlim([6e-8,1])
            count = count+1
        
        plt.legend(loc = 'upper left', bbox_to_anchor=(0., 0, 0, 0.95))
        plt.tight_layout()
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_boxplot_pq.pdf'),dpi=300,bbox_inches='tight')
        plt.show()
        
        
    def describeAdoption(self):
        if self.state=='wa':
            colorb = 'blue'; colorp = 'blue'
            colors = ['#81dbed','#60a3f2','blue'] 
            cm = ['blue', 'blue', 'blue']
            county = 'King'
        else:
            colorb = 'orange'; colorp = 'brown'
            colors = ['#ffc34b', '#ff6e00', '#ff0000']
            cm = ['orange', 'orange', 'orange']
            county = 'Los Angeles'
        
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        # 2.1 Distribution of prediction errors (DE/ABM, tract-average, 2010-2022)
        data_error = data.copy()
        data_error['State'] = ((data_error['DS_12']-data_error['EMP_12'])/data_error['POPULATION']).fillna(0).clip(-1, 1)*100
        data_error['County'] = ((data_error['DE_12']-data_error['EMP_12'])/data_error['POPULATION']).fillna(0).clip(-1, 1)*100
        data_error['SocNet'] = ((data_error['ABM_12']-data_error['EMP_12'])/data_error['POPULATION']).fillna(0).clip(-1, 1)*100
        print('error',data_error['State'].mean(),data_error['County'].mean(),data_error['SocNet'].mean())
        print('error',data_error['State'].abs().mean(),data_error['County'].abs().mean(),data_error['SocNet'].abs().mean())
        print('error',data_error['State'].abs().std(),data_error['County'].abs().std(),data_error['SocNet'].abs().std())
        
        print('error',data_error['State'].mean(),data_error['County'].mean(),data_error['SocNet'].mean())
        
        fig = plt.figure(figsize=(4.2,1.5))
        axs = fig.add_subplot()
        data_error = pd.melt(data_error, id_vars=['GEOID'], value_vars=['State','County','SocNet'], var_name='Model', value_name='Error')
        gfg = sns.histplot(data = data_error, x="Error", hue="Model", stat="percent", bins = 80, multiple="dodge", shrink=.8, edgecolor='k',linewidth=1, palette=cm)
        hatches = ['x','*','o']
        for container, hatch, handle in zip(gfg.containers, hatches, gfg.get_legend().legend_handles[::-1]):
            handle.set_hatch(hatch)
            for rectangle in container:
                rectangle.set_hatch(hatch)
        
        sns.move_legend(gfg, "upper left", ncol=3)
        gfg.legend_.set_title(None)
        plt.xlim(-6, 6)
        plt.ylim(0, 30)
        plt.xlabel('Adoption Rate Error [%]')
        plt.ylabel('Percentage [%]')
        plt.axvline(0, color=".3", dashes=(2, 2))
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join(self.FIGURE_PATH, 'now_hist_tract.pdf'),dpi=300,bbox_inches='tight')
        plt.show()
        
        
        data_error = data.copy()
        data_error['State'] = ((data_error['DS_12'])/data_error['POPULATION']).fillna(0).clip(-1, 1)*100
        data_error['County'] = ((data_error['DE_12'])/data_error['POPULATION']).fillna(0).clip(-1, 1)*100
        data_error['SocNet'] = ((data_error['ABM_12'])/data_error['POPULATION']).fillna(0).clip(-1, 1)*100
        data_error['Data'] = ((data_error['EMP_12'])/data_error['POPULATION']).fillna(0).clip(-1, 1)*100
        data_error = pd.pivot_table(data_error, values=['State','County','SocNet','Data'], index=['GEOID'], aggfunc="mean", fill_value=0).reset_index()
        
        fig = plt.figure(figsize=(2.8,3))
        axs = fig.add_subplot()
        sns.scatterplot(ax = axs, data=data_error, x='Data', y='SocNet',c='#81dbed',s=3, label = 'SocNet')
        sns.scatterplot(ax = axs, data=data_error, x='Data', y='County',c='blue',s=3, label = 'County')
        sns.scatterplot(ax = axs, data=data_error, x='Data', y='State',c='orange',s=3, label = 'State')
        axs.axline((1, 1), slope=1, c = 'k', linestyle='dashed')
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        lgnd = plt.legend(markerscale=5)
        plt.xlabel('Adoption Rate of Data [%]')
        plt.ylabel('Adoption Rate of Model [%]')
        plt.savefig(os.path.join(self.FIGURE_PATH, 'si_scatter_error.png'),dpi=300,bbox_inches='tight')


        # #2.2 Temporal: Curves of cumulative adoptions (DE/ABM/EMP, 3 groups, 2010-2022)
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        data_temp = data.copy()
        abm_names=[]; de_names=[]; emp_names=[]; ds_names=[]
        for i in range(13):
            abm_names.append('ABM_'+str(i))
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        
        data_temp_low = data_temp[data_temp['CLASS']=='Low']
        data_temp_mid = data_temp[data_temp['CLASS']=='Middle']
        data_temp_high = data_temp[data_temp['CLASS']=='High']
        
        data_temp_low = pd.pivot_table(data_temp_low, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_temp_mid = pd.pivot_table(data_temp_mid, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_temp_high = pd.pivot_table(data_temp_high, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_total = pd.pivot_table(data_temp, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        
        pop_low = data_temp_low.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        pop_mid = data_temp_mid.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        pop_high = data_temp_high.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        pop_total = data_total.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        
        low_b = 5; mid_b = 50; high_b = 95
        print('MAE')
        print("SocNet")
        dtl = data_total
        h_m = np.percentile(dtl[abm_names].values,mid_b,axis=0)/pop_total
        h_d = np.percentile(dtl[emp_names].values,mid_b,axis=0)/pop_total
        print(h_m,h_d)
        print(np.mean(np.abs((h_m-h_d))))
        print("State")
        dtl = data_total
        h_m = np.percentile(dtl[ds_names].values,mid_b,axis=0)/pop_total
        h_d = np.percentile(dtl[emp_names].values,mid_b,axis=0)/pop_total
        print(h_m,h_d)
        print(np.mean(np.abs((h_m-h_d))))
        print("County")
        dtl = data_total
        h_m = np.percentile(dtl[de_names].values,mid_b,axis=0)/pop_total
        h_d = np.percentile(dtl[emp_names].values,mid_b,axis=0)/pop_total
        print(h_m,h_d)
        print(np.mean(np.abs((h_m-h_d))))

        fig = plt.figure(figsize=(2.3,3))
        axs = fig.add_subplot()
        dtl = data_temp_low
        p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
        p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
        p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
        plt.plot(np.arange(2010,2023,2),p_50[::2]/pop_low, color = colors[0], alpha= 0.5,marker='o')
        plt.plot(np.arange(2010,2023,2),np.percentile(dtl[de_names].values,mid_b,axis=0)[::2]/pop_low, color = colors[0], alpha= 0.5,marker='*', markersize=10 )
        plt.plot(np.arange(2010,2023,2),np.percentile(dtl[ds_names].values,mid_b,axis=0)[::2]/pop_low, color = colors[0], alpha= 0.5, marker='x',markersize=10)
        plt.scatter(np.arange(2010,2023),np.percentile(dtl[emp_names].values,mid_b,axis=0)/pop_low,  color = colors[0], alpha= 0.5, marker='s' )

        dtl = data_temp_mid
        p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
        p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
        p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
        plt.plot(np.arange(2010,2023,2),p_50[::2]/pop_mid, color = colors[1], alpha= 0.5, marker='o')
        plt.plot(np.arange(2010,2023,2),np.percentile(dtl[de_names].values,mid_b,axis=0)[::2]/pop_mid, color = colors[1], alpha= 0.5,marker='*', markersize=10)
        plt.plot(np.arange(2010,2023,2),np.percentile(dtl[ds_names].values,mid_b,axis=0)[::2]/pop_mid, color = colors[1], alpha= 0.5,marker='x',markersize=10)
        plt.scatter(np.arange(2010,2023),np.percentile(dtl[emp_names].values,mid_b,axis=0)/pop_mid,  color = colors[1], alpha= 0.5, marker='s' )

        dtl = data_temp_high
        p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
        p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
        p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
        plt.plot(np.arange(2010,2023,2),np.percentile(dtl[ds_names].values,mid_b,axis=0)[::2]/pop_high, color = colors[2], alpha= 0.5, marker='x',label = 'State',markersize=10)
        plt.plot(np.arange(2010,2023,2),np.percentile(dtl[de_names].values,mid_b,axis=0)[::2]/pop_high, color = colors[2], alpha= 0.5, marker='*', label = 'County',markersize=10)
        plt.plot(np.arange(2010,2023,2),p_50[::2]/pop_high, color = colors[2], alpha= 0.5, marker='o', label = 'SocNet')
        plt.scatter(np.arange(2010,2023,2),np.percentile(dtl[emp_names].values,mid_b,axis=0)[::2]/pop_high,  color = colors[2], alpha= 0.5,  marker='s',label = 'Data')
        
        plt.xlabel('Year')
        plt.ylabel('Adoption Rate [%]')
        plt.legend(ncols=1)
        plt.tight_layout()
        
        plt.tight_layout()
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH, 'now_curve_state.pdf'),dpi=300,bbox_inches='tight')
        plt.show()
        
        # 2.4 boxplot of counties
        data_emp = data.copy()
        df_emp = data_emp.drop_duplicates(subset=['GEOID'])
        df_emp = pd.pivot_table(df_emp, values=de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY'], aggfunc="sum", fill_value=0).reset_index()
        df_emp = df_emp.sort_values(by='POPULATION',ascending=False)
        county_names = df_emp.sort_values(by='POPULATION',ascending=False)['COUNTY'].values
        interval_1 = 15; interval_2 = int((len(county_names)-interval_1)/2)+interval_1
        
        data_abm = data.copy()
        data_abm = pd.pivot_table(data_abm, values=abm_names+['POPULATION'], index=['COUNTY','REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_abm = data_abm.sort_values(by='POPULATION',ascending=False)
        
        df_emp['DS_12'] = df_emp['DS_12']/df_emp['POPULATION']*100
        df_emp['DE_12'] = df_emp['DE_12']/df_emp['POPULATION']*100
        df_emp['EMP_12'] = df_emp['EMP_12']/df_emp['POPULATION']*100
        data_abm['ABM_12'] = data_abm['ABM_12']/data_abm['POPULATION']*100
        
        fig, ax = plt.subplots(figsize=(9,1.8))
        sns.boxplot(ax = ax, x='COUNTY', y='ABM_12', data=data_abm[data_abm.COUNTY.isin(county_names[:interval_1])], width=.4, orient='x', palette = cycle([colorb]),fill=False,showfliers = False)
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[:interval_1])]['DS_12'].values, x=[i + (1) * 0.4 for i in range(interval_1)], color='k',marker='x',label='State')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[:interval_1])]['DE_12'].values, x=[i + (1) * 0.4 for i in range(interval_1)], color='k',marker='*',label='County',s=200)
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[:interval_1])]['EMP_12'].values, x=[i + (1) * 0.4 for i in range(interval_1)], color='k',marker='s',label='Data',s= 40)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45 )
        plt.ylabel('Adoption Rate [%]')
        plt.xlabel('')
        plt.ylim([0,5])
        plt.legend(loc = 'upper right', ncol=3)
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_boxplot_county_1.pdf'),dpi=300,bbox_inches='tight')
        plt.show()


        fig, ax = plt.subplots(figsize=(12,1.8))
        sns.boxplot(ax = ax, x='COUNTY', y='ABM_12', data=data_abm[data_abm.COUNTY.isin(county_names[interval_1:interval_2])], palette=cycle([colorb]), width=.4, orient="x",fill=False,showfliers = False, legend='full')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_1:interval_2])]['DS_12'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_1:interval_2]))], color='k',marker='x',label='State')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_1:interval_2])]['DE_12'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_1:interval_2]))], color='k',marker='*',label='County',s=200)
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_1:interval_2])]['EMP_12'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_1:interval_2]))], color='k',marker='s',label='Data',s=40)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45 )
        plt.ylabel('Adoption Rate [%]')
        plt.xlabel('')
        plt.ylim([0,6])
        plt.legend(loc = 'upper right', ncol=3)
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_boxplot_county_2.pdf'),dpi=300,bbox_inches='tight')
        plt.show()


        fig, ax = plt.subplots(figsize=(12,1.8))
        sns.boxplot(ax = ax, x='COUNTY', y='ABM_12', data=data_abm[data_abm.COUNTY.isin(county_names[interval_2:])], palette = cycle([colorb]), width=.4, orient="x",fill=False,showfliers = False, legend='full')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_2:])]['DS_12'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_2:]))], color='k',marker='x',label='State')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_2:])]['DE_12'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_2:]))], color='k',marker='*',label='County',s= 200)
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_2:])]['EMP_12'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_2:]))], color='k',marker='s',label='Data',s = 40)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45 )
        plt.ylabel('Adoption Rate [%]')
        plt.xlabel('')
        plt.ylim([0,6])
        plt.legend(loc = 'upper right', ncol=3)
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'now_boxplot_county_3.pdf'),dpi=300,bbox_inches='tight')
        plt.show()


    def futureAdoption(self):        
        if self.state == 'wa':
            cm = bcmap; vm = 15; cb = 'blue'; colorb = 'blue'
        else:
            cm = 'YlOrRd'; vm = 10; cb = 'orange'; colorb = 'orange'
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        # # 4.1 future curve
        data_temp = data.copy()
        end = 41
        abm_names=[]; de_names=[]; emp_names=[]; ds_names = []
        for i in range(end):
            abm_names.append('ABM_'+str(i))
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))
        
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        
        data_temp_low = data_temp[data_temp['CLASS']=='Low']
        data_temp_mid = data_temp[data_temp['CLASS']=='Middle']
        data_temp_high = data_temp[data_temp['CLASS']=='High']
        
        data_temp_low = pd.pivot_table(data_temp_low, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_temp_mid = pd.pivot_table(data_temp_mid, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_temp_high = pd.pivot_table(data_temp_high, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_total = pd.pivot_table(data_temp, values=abm_names+de_names+emp_names+ds_names+['POPULATION'], index=['REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        
        pop_low = data_temp_low.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        pop_mid = data_temp_mid.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        pop_high = data_temp_high.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        pop_total = data_total.drop_duplicates(subset=['POPULATION'])['POPULATION'].sum()/100
        
        low_b = 5; mid_b = 50; high_b = 95
        fig = plt.figure(figsize=(2.3,2)) 
        axs = fig.add_subplot()

        dtl = data_total
        p_5 = np.percentile(dtl[abm_names].values,low_b,axis=0)                
        p_50 = np.percentile(dtl[abm_names].values,mid_b,axis=0)
        p_95 = np.percentile(dtl[abm_names].values,high_b,axis=0)
        plt.plot(np.arange(2010,2010+end),np.percentile(dtl[ds_names].values,mid_b,axis=0)/pop_total, color = cb, alpha= 0.5, marker='x', label = 'State' , markersize=10,markevery=5)
        plt.plot(np.arange(2010,2010+end),np.percentile(dtl[de_names].values,mid_b,axis=0)/pop_total, color = cb, alpha= 0.5, marker='*', label = 'County', markersize=10, markevery=5)
        plt.plot(np.arange(2010,2010+end),p_50/pop_total, color = cb, alpha= 0.5, marker='o', label = 'SocNet', markevery=5)

        plt.xlabel('Year')
        plt.ylabel('Adoption Rate [%]')
        plt.legend(ncols=1)
        plt.tight_layout()
        
        plt.tight_layout()
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH, 'future_curve_state.pdf'),dpi=300,bbox_inches='tight')
        plt.show()
            
        data_emp = data.copy()
        df_emp = data_emp.drop_duplicates(subset=['GEOID'])
        df_emp = pd.pivot_table(df_emp, values=de_names+emp_names+ds_names+['POPULATION'], index=['COUNTY'], aggfunc="sum", fill_value=0).reset_index()
        df_emp = df_emp.sort_values(by='POPULATION',ascending=False)
        county_names = df_emp.sort_values(by='POPULATION',ascending=False)['COUNTY'].values
        interval_1 = 15; interval_2 = int((len(county_names)-interval_1)/2)+interval_1
        
        data_abm = data.copy()
        data_abm = pd.pivot_table(data_abm, values=abm_names+['POPULATION'], index=['COUNTY','REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_abm = data_abm.sort_values(by='POPULATION',ascending=False)
        
        df_emp['DS_40'] = df_emp['DS_40']/df_emp['POPULATION']*100
        df_emp['DE_40'] = df_emp['DE_40']/df_emp['POPULATION']*100
        data_abm['ABM_40'] = data_abm['ABM_40']/data_abm['POPULATION']*100
        
        fig, ax = plt.subplots(figsize=(7,1.6))
        sns.boxplot(ax = ax, x='COUNTY', y='ABM_40', data=data_abm[data_abm.COUNTY.isin(county_names[:interval_1])], width=.4, orient='x', palette = cycle([colorb]),fill=False,showfliers = False)
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[:interval_1])]['DS_40'].values, x=[i + (1) * 0.4 for i in range(interval_1)], color='k',marker='x',label='State')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[:interval_1])]['DE_40'].values, x=[i + (1) * 0.4 for i in range(interval_1)], color='k',marker='*',label='County',s=200)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        plt.ylabel('Adoption Rate [%]')
        plt.xlabel('')

        plt.ylim([0,100])
        if self.state == 'ca':
            plt.legend(loc = 'upper left', ncol= 2)
        else:
            plt.legend('')
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'future_boxplot_county_1.pdf'),dpi=300,bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(figsize=(12,1.6))
        sns.boxplot(ax = ax, x='COUNTY', y='ABM_40', data=data_abm[data_abm.COUNTY.isin(county_names[interval_1:interval_2])], palette=cycle([colorb]), width=.4, orient="x",fill=False,showfliers = False, legend='full')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_1:interval_2])]['DS_40'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_1:interval_2]))], color='k',marker='x',label='State')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_1:interval_2])]['DE_40'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_1:interval_2]))], color='k',marker='*',label='County',s=200)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        plt.ylabel('Adoption Rate [%]')
        plt.xlabel('')
        plt.ylim([0,100])
        if self.state == 'ca':
            plt.legend(loc = 'upper left', ncol= 2)
        else:
            plt.legend('')
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'future_boxplot_county_2.pdf'),dpi=300,bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(figsize=(12,1.6))
        sns.boxplot(ax = ax, x='COUNTY', y='ABM_40', data=data_abm[data_abm.COUNTY.isin(county_names[interval_2:])], palette = cycle([colorb]), width=.4, orient="x",fill=False,showfliers = False, legend='full')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_2:])]['DS_40'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_2:]))], color='k',marker='x',label='State')
        sns.scatterplot(ax = ax, y=df_emp[df_emp.COUNTY.isin(county_names[interval_2:])]['DE_40'].values, x=[i + (1) * 0.4 for i in range(len(county_names[interval_2:]))], color='k',marker='*',label='County',s= 200)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        plt.ylabel('Adoption Rate [%]')
        plt.xlabel('')
        plt.ylim([0,100])
        if self.state == 'ca':
            plt.legend(loc = 'upper left', ncol= 2)
        else:
            plt.legend('')
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'future_boxplot_county_3.pdf'),dpi=300,bbox_inches='tight')
        plt.show()

        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'future_boxplot_county.pdf'),dpi=300,bbox_inches='tight')

    def generateSI(self):
        if self.state == 'wa':
            cm = bcmap; vm = 15; cb = '#00429d'; name = 'NAME20'; city_name = 'Seattle'
        else:
            cm = 'YlOrRd'; vm = 10; cb = 'orange'; name = 'NAME'; city_name = 'Los Angeles'
        
        # 1 Map of p,q values (ABM, 3 groups, 2010-2022). Since there are 3 groups, there are 6 maps here.
        pq = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_pq.csv'))
        pq_geo = pq.copy()
        result = self.county_shp.merge(pq_geo,right_on='COUNTY',left_on='NAME')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        
        fig = plt.figure(figsize=(10,6))
        axs = fig.add_subplot(2, 3, 1)
        db_fil = db[db['p_low_abm']!=-1]
        ax = db_fil.plot(ax = axs, column="p_low_abm", cmap='OrRd', vmax=0.003, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='p')
        plt.title('SNM, Low Income, p')
        axs = fig.add_subplot(2, 3, 2)
        db_fil = db[db['p_middle_abm']!=-1]
        ax = db_fil.plot(ax = axs, column="p_middle_abm", cmap='OrRd',vmax=0.003,  edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='p')
        plt.title('SNM, Middle Income, p')
        axs = fig.add_subplot(2, 3, 3)
        db_fil = db[db['p_high_abm']!=-1]
        ax = db_fil.plot(ax = axs, column="p_high_abm", cmap='OrRd',vmax=0.003, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='p')
        plt.title('SNM, High Income, p')
        axs = fig.add_subplot(2, 3, 4)
        db_fil = db[db['p_low_de']!=-1]
        ax = db_fil.plot(ax = axs, column="p_low_de", cmap='OrRd', vmax=0.001, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='p')
        plt.title('CBM, Low Income, p')
        axs = fig.add_subplot(2, 3, 5)
        db_fil = db[db['p_middle_de']!=-1]
        ax = db_fil.plot(ax = axs, column="p_middle_de", cmap='OrRd',vmax=0.001, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='p')
        plt.title('CBM, Middle Income, p')
        axs = fig.add_subplot(2, 3, 6)
        db_fil = db[db['p_high_de']!=-1]
        ax = db_fil.plot(ax = axs, column="p_high_de", cmap='OrRd',vmax=0.001, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='p')
        plt.title('CBM, High Income, p')
        plt.tight_layout()
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'si_map_p.png'))
        plt.show()
        
        fig = plt.figure(figsize=(10,6))
        axs = fig.add_subplot(2, 3, 1)
        db_fil = db[db['q_low_abm']!=-1]
        ax = db_fil.plot(ax = axs, column="q_low_abm", cmap='OrRd', vmax=1,  edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='q')
        plt.title('SNM, Low Income, q')
        axs = fig.add_subplot(2, 3, 2)
        db_fil = db[db['q_middle_abm']!=-1]
        ax = db_fil.plot(ax = axs, column="q_middle_abm", cmap='OrRd',vmax=1,  edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='q')
        plt.title('SNM, Middle Income, q')
        axs = fig.add_subplot(2, 3, 3)
        db_fil = db[db['q_high_abm']!=-1]
        ax = db_fil.plot(ax = axs, column="q_high_abm", cmap='OrRd',vmax=1,  edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='q')
        plt.title('SNM, High Income, q')
        axs = fig.add_subplot(2, 3, 4)
        db_fil = db[db['q_low_de']!=-1]
        ax = db_fil.plot(ax = axs, column="q_low_de", cmap='OrRd', vmax=0.5, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='q')
        plt.title('CBM, Low Income, q')
        axs = fig.add_subplot(2, 3, 5)
        db_fil = db[db['q_middle_de']!=-1]
        ax = db_fil.plot(ax = axs, column="q_middle_de", cmap='OrRd',vmax=0.5, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='q')
        plt.title('CBM, Middle Income, q')
        axs = fig.add_subplot(2, 3, 6)
        db_fil = db[db['q_high_de']!=-1]
        ax = db_fil.plot(ax = axs, column="q_high_de", cmap='OrRd',vmax=0.5, edgecolor=None, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.collections[0]
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='q')
        plt.title('CBM, High Income, q')
        plt.tight_layout()
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH, 'si_map_q.png'))
        plt.show()
        
        # # 2 Map of cumulative adoptions (DE/ABM, 2035)
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})
        end = 40
        abm_names=[]; de_names=[]; emp_names=[]; ds_names = []
        for i in range(end):
            abm_names.append('ABM_'+str(i))
        for i in range(end):
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))
        for i in range(13):
            emp_names.append('EMP_'+str(i))
        data_map = data.copy()
        data_map = data_map[data_map['POPULATION']>0]
        data_map = pd.pivot_table(data_map, values=abm_names+de_names+ds_names+['POPULATION'], index=['COUNTY','REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_map_abm = pd.pivot_table(data_map, values=abm_names+['POPULATION'], index=['COUNTY'], aggfunc="mean", fill_value=0).reset_index()
        data_map_de = pd.pivot_table(data_map, values=de_names+['POPULATION'], index=['COUNTY'], aggfunc="mean", fill_value=0).reset_index()
        data_map_ds = pd.pivot_table(data_map, values=ds_names+['POPULATION'], index=['COUNTY'], aggfunc="mean", fill_value=0).reset_index()
        data_map_de['DE_'+str(end-1)] = data_map_de['DE_'+str(end-1)]/(data_map_de['POPULATION'])*100
        data_map_abm['ABM_'+str(end-1)] = data_map_abm['ABM_'+str(end-1)]/(data_map_abm['POPULATION'])*100
        data_map_ds['DS_'+str(end-1)] = data_map_ds['DS_'+str(end-1)]/(data_map_ds['POPULATION'])*100
        
        result = self.county_shp.merge(data_map_ds,right_on='COUNTY',left_on='NAME')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        fig = plt.figure(figsize=(7,3))
        axs = fig.add_subplot(1, 3, 1)
        ax = db.plot(ax = axs, column='DS_'+str(end-1), cmap=cm, vmin=0, vmax=vm,edgecolor=None, legend=False,linewidth=0.0)
        scatter = ax.collections[0]
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Adoption Rate [%]')
        plt.title('State BM')
        
        result = self.county_shp.merge(data_map_de,right_on='COUNTY',left_on='NAME')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        axs = fig.add_subplot(1, 3, 2)
        ax = db.plot(ax = axs, column='DE_'+str(end-1), cmap=cm, vmin=0, vmax=vm, edgecolor=None, legend=False,linewidth=0.0)
        scatter = ax.collections[0]
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Adoption Rate [%]')
        plt.title('County BM')
        
        result = self.county_shp.merge(data_map_abm,right_on='COUNTY',left_on='NAME')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        axs = fig.add_subplot(1, 3, 3)
        ax = db.plot(ax = axs, column='ABM_'+str(end-1), cmap=cm,vmin=0, vmax=vm,edgecolor=None, legend=False,linewidth=0.0)
        scatter = ax.collections[0]
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Adoption Rate [%]')
        plt.title('SocNet BM')
        plt.tight_layout()
        
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH, 'si_map_future.png'))
        
        # 3. Moran future map
        end = 40
        abm_names = []; de_names = []; emp_names= []; ds_names= []
        for i in range(end):
            abm_names.append('ABM_'+str(i))
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))

        data_emp = data.copy(); 
        data_emp = data_emp[data_emp['POPULATION']>0]
        data_emp = data_emp.drop_duplicates(subset=['GEOID','CLASS'])
        data_emp = pd.pivot_table(data_emp, values=de_names+ds_names+['POPULATION'], index=['GEOID','COUNTY'], aggfunc="sum", fill_value=0).reset_index()
        
        city = self.place_shp[self.place_shp[name]==city_name]
        city = gpd.GeoDataFrame(city, crs=city.crs).to_crs(epsg=3857)
        
        db = self.tract_shp.merge(data_emp,right_on='GEOID',left_on='GEOID',how='left').fillna(0)
        db = gpd.GeoDataFrame(db, crs=db.crs).to_crs(epsg=3857)
        w = weights.KNN.from_dataframe(db, k=8); w.transform = "R"

        moran_emp = esda.moran.Moran(db['DS_'+str(end-1)], w)
        print(moran_emp.I,moran_emp.p_sim,moran_emp.z_sim)
        moran_emp_loc = Moran_Local(db['DS_'+str(end-1)], w)
        lisa_cluster(city, moran_emp_loc, db, p=0.05, figsize = (4,4),legend_kwds={'loc': 'upper right', 'bbox_to_anchor': (1.5, 1)})
        plt.title('State BM')
        plt.show()
        
        moran_emp = esda.moran.Moran(db['DE_'+str(end-1)], w)
        print(moran_emp.I,moran_emp.p_sim,moran_emp.z_sim)
        moran_emp_loc = Moran_Local(db['DE_'+str(end-1)], w)
        lisa_cluster(city, moran_emp_loc, db, p=0.05, figsize = (4,4),legend_kwds={'loc': 'upper right', 'bbox_to_anchor': (1.5, 1)})
        plt.title('County BM')
        plt.show()
        
        data_abm = data.copy()
        data_abm = data_abm[data_abm['POPULATION']>0]
        data_abm = pd.pivot_table(data_abm, values=abm_names+['POPULATION'], index=['COUNTY','GEOID','REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_abm = pd.pivot_table(data_abm, values=abm_names+['POPULATION'], index=['COUNTY','GEOID'], aggfunc="mean", fill_value=0).reset_index()
        db = self.tract_shp.merge(data_abm,right_on='GEOID',left_on='GEOID',how='left').fillna(0)
        db = gpd.GeoDataFrame(db, crs=db.crs).to_crs(epsg=3857)
        w = weights.KNN.from_dataframe(db, k=8); w.transform = "R"
        
        moran_abm = esda.moran.Moran(db['ABM_'+str(end-1)], w)
        print(moran_abm.I,moran_abm.p_sim,moran_abm.z_sim)
        moran_abm_loc = Moran_Local(db['ABM_'+str(end-1)], w)
        lisa_cluster(city, moran_abm_loc, db, p=0.05, figsize = (4,4),legend_kwds={'loc': 'upper right', 'bbox_to_anchor': (1.5, 1)})
        plt.title('SocNet BM')
        plt.show()

        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH,'si_map_moran.png'))
        
    def FutureTractMap(self,city_name):
        if self.state == 'wa':
            cm = bcmap; vm = 30; name = 'NAME20'
        else:
            cm = 'YlOrRd'; vm = 10; name = 'NAME'
        data = pd.read_csv(os.path.join( '..', 'data', self.state, self.state+'_data.csv'),converters={'GEOID': str})

        end  = 20
        abm_names=[]; de_names=[]; ds_names = []
        for i in range(end):
            abm_names.append('ABM_'+str(i))
        for i in range(end):
            de_names.append('DE_'+str(i))
            ds_names.append('DS_'+str(i))
            
        data_map = data.copy()
        data_map = data_map[data_map['POPULATION']>0]
        data_map = pd.pivot_table(data_map, values=abm_names+de_names+ds_names+['POPULATION','INCOME'], index=['GEOID','REP','BATCH'], aggfunc="sum", fill_value=0).reset_index()
        data_map_abm = pd.pivot_table(data_map, values=abm_names+['POPULATION'], index=['GEOID'], aggfunc="mean", fill_value=0).reset_index()
        data_map_de = pd.pivot_table(data_map, values=de_names+['POPULATION'], index=['GEOID'], aggfunc="mean", fill_value=0).reset_index()
        data_map_ds = pd.pivot_table(data_map, values=ds_names+['POPULATION'], index=['GEOID'], aggfunc="mean", fill_value=0).reset_index()
        data_map_income = pd.pivot_table(data_map, values=['INCOME'], index=['GEOID'], aggfunc="mean", fill_value=0).reset_index()
        data_map_income['INCOME'] = data_map_income['INCOME']/1000
        data_map_de['DE_'+str(end-1)] = data_map_de['DE_'+str(end-1)]/(data_map_de['POPULATION'])*100
        data_map_abm['ABM_'+str(end-1)] = data_map_abm['ABM_'+str(end-1)]/(data_map_abm['POPULATION'])*100
        data_map_ds['DS_'+str(end-1)] = data_map_ds['DS_'+str(end-1)]/(data_map_ds['POPULATION'])*100
        
        ## 5.1 Map of cumulative adoption
        city = self.place_shp[self.place_shp[name]==city_name]
        city = gpd.GeoDataFrame(city, crs=city.crs).to_crs(epsg=3857)
        result = self.tract_shp.merge(data_map_ds,right_on='GEOID',left_on='GEOID')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        fig = plt.figure(figsize=(7,3))
        axs = fig.add_subplot(1, 3, 1)
        db = gpd.clip(db, city)
        ax = db.plot(ax = axs, column='DS_'+str(end-1), cmap=cm, vmin=0, vmax=vm,edgecolor=None, legend=False,linewidth=0.0)
        # ax = db.plot(ax = axs, column='DS_'+str(end-1), cmap=cm, vmin=0, edgecolor=None, legend=False,linewidth=0.0)
        scatter = ax.collections[0]
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Adoption Rate [%]')
        plt.title('State BM')
        
        result = self.tract_shp.merge(data_map_de,right_on='GEOID',left_on='GEOID')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        axs = fig.add_subplot(1, 3, 2)
        db = gpd.clip(db, city)
        ax = db.plot(ax = axs, column='DE_'+str(end-1), cmap=cm, vmin=0, vmax=vm, edgecolor=None, legend=False,linewidth=0.0)
        # ax = db.plot(ax = axs, column='DE_'+str(end-1), cmap=cm, vmin=0,  edgecolor=None, legend=False,linewidth=0.0)
        scatter = ax.collections[0]
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Adoption Rate [%]')
        plt.title('County BM')
        
        result = self.tract_shp.merge(data_map_abm,right_on='GEOID',left_on='GEOID')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        axs = fig.add_subplot(1, 3, 3)
        db = gpd.clip(db, city)
        ax = db.plot(ax = axs, column='ABM_'+str(end-1), cmap=cm,vmin=0, vmax=vm,edgecolor=None, legend=False,linewidth=0.0)
        # ax = db.plot(ax = axs, column='ABM_'+str(end-1), cmap=cm,vmin=0, edgecolor=None, legend=False,linewidth=0.0)
        scatter = ax.collections[0]
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Adoption Rate [%]')
        plt.title('SocNet BM')
        plt.tight_layout()
        
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH, 'map_future_city.png'))
        
        ## 5.2 Map of income        
        city = self.place_shp[self.place_shp[name]==city_name]
        city = gpd.GeoDataFrame(city, crs=city.crs).to_crs(epsg=3857)
        result = self.tract_shp.merge(data_map_income,right_on='GEOID',left_on='GEOID')
        db = gpd.GeoDataFrame(result, crs=result.crs).to_crs(epsg=3857)
        fig = plt.figure(figsize=(3,3))
        axs = fig.add_subplot(1, 1, 1)
        db = gpd.clip(db, city)
        ax = db.plot(ax = axs, column='INCOME', cmap=cm, vmin=0, edgecolor=None, legend=False,linewidth=0.0)
        # ax = db.plot(ax = axs, column='DS_'+str(end-1), cmap=cm, vmin=0, edgecolor=None, legend=False,linewidth=0.0)
        scatter = ax.collections[0]
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.4, label='Income [1,000 $]')
        plt.title(city_name)
        plt.tight_layout()
        
        if not os.path.exists(os.path.join( self.FIGURE_PATH)):
            os.makedirs(os.path.join( self.FIGURE_PATH))
        plt.savefig(os.path.join( self.FIGURE_PATH, 'map_future_income.png'))


if __name__ == "__main__":
    state = 'wa'; county = 'King'
    state_county = pd.read_csv(os.path.join('..','data', state, 'fips-by-state.csv'),encoding= 'unicode_escape')
    state_county = state_county[state_county['state']=='WA']
    counties = state_county.name.str.split(' County',expand=True)[0].unique()
    pdt = VisData(state, counties)
    pdt.loadData()
    pdt.currentAdoption()
    pdt.analyzeAdoption()
    pdt.describeAdoption()
    pdt.futureAdoption()
    
    state = 'ca'; county = 'Los Angeles'
    state_county = pd.read_csv(os.path.join('..','data', state, 'fips-by-state.csv'), encoding= 'unicode_escape')
    state_county = state_county[state_county['state']=='CA']
    counties = state_county.name.str.split(' County',expand=True)[0].unique()
    pdt = VisData(state, counties)
    pdt.loadData()
    pdt.currentAdoption()
    pdt.analyzeAdoption()
    pdt.describeAdoption()
    pdt.futureAdoption()