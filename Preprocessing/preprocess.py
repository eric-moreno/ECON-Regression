import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

def prepare_data(df_wsimenergy, df):
    """
    Take in econ data frame and select out the data for the \
    top 50 most common ids
    """
    
    print(df_wsimenergy)
    #Another dataframe to perform counting 
    df_mask = df_wsimenergy[['layer','id','index']].droplevel(1)
    df_mask = df_mask.groupby(['layer','id']).count()
    df_mask['layer'] = df_mask.index.get_level_values('layer')
    df_mask['id'] = df_mask.index.get_level_values('id')
    
    #Count
    df_mask['count'] = df_mask['index']/16
    df_mask = df_mask.drop(['index'], axis = 1)
    
    #Select the ids
    print(df_mask.sort_values(['count'], ascending = False)[:50])
    id_list = df_mask.sort_values(['count'], ascending = False).iloc[:50]['id'].tolist()
    
    #return the new dataframe with only the selected ids
    return df[df['id'].isin(id_list)], id_list


files = os.listdir('../data/crab_gun_electrons_01_25_22/220126_181410/0000/')

df_tc_individual = []
df_econ_individual = []
df_gen_individual = []

for i in range(len(files[:1])): 
    file = files[i]
    print(file)
    
    fname = "../data/crab_gun_electrons_01_25_22/220126_181410/0000/" + file
    ev_dict = uproot.open(fname)["FloatingpointAutoEncoderStrideDummyHistomaxGenmatchGenclustersntuple/HGCalTriggerNtuple"]
    
    arrays_toread = [
    "econ_index","econ_data",
    #"econ_subdet","econ_zside",
    "econ_layer","econ_waferu","econ_waferv","econ_wafertype",
    "tc_simenergy",
    #"tc_subdet","tc_zside",
    "tc_layer","tc_waferu","tc_waferv","tc_wafertype",
    #"gen_pt","gen_energy","gen_eta","gen_phi",
    #"genpart_pt","genpart_energy",
    "econ_id", "tc_id"
    ]
    events = ev_dict.arrays(arrays_toread)

    #Separate the data sets
    econ = ak.zip({
        "index": events['econ_index'],
        "id":events['econ_id'],
        "data": events["econ_data"],
        #"subdet": events["econ_subdet"],
        #"zside": events["econ_zside"],
        "layer": events["econ_layer"],
        "waferu": events["econ_waferu"],
        "waferv": events["econ_waferv"],
    })

    tc = ak.zip({
        "simenergy": events["tc_simenergy"],
        #"subdet": events["tc_subdet"],
        #"zside": events["tc_zside"],
        "layer": events["tc_layer"],
        "waferu": events["tc_waferu"],
        "waferv": events["tc_waferv"],
    })
    del events
    
    #gen = ak.zip({
    #    "pt": events["gen_pt"],
    #    "energy": events["gen_energy"],
    #    "eta": events["gen_eta"],
    #    "phi": events["gen_phi"],
    #})
    
    temp_tc = ak.to_pandas(tc).reset_index()#.drop(columns=['subdet', 'zside', 'level_0'])
    del tc
    temp_econ = ak.to_pandas(econ).reset_index()#.drop(columns=['subdet', 'zside', 'level_0'])
    del econ
    
    #temp_gen = ak.to_pandas(gen).reset_index().drop(columns=['subdet', 'zside', 'level_0'])
    
    temp_tc['entry'] = (temp_tc['entry'].to_numpy()+ i*1000)
    temp_econ['entry'] = (temp_econ['entry'].to_numpy()+ i*1000)
    #temp_gen['entry'] = (temp_gen['entry'].to_numpy()+ i*1000)
    
    df_tc_individual.append(temp_tc)
    df_econ_individual.append(temp_econ)
    #df_gen_individual.append(temp_gen)

df_tc = pd.concat(df_tc_individual)
del df_tc_individual
df_econ = pd.concat(df_econ_individual)
del df_econ_individual
#df_gen = pd.concat(df_gen_individual)
#del df_gen_individual


df_simtotal = df_tc.groupby(['entry','layer','waferu','waferv'])["simenergy"].sum()
#Prepare df_econ
df_econ.reset_index(inplace=True)
df_econ.set_index(['entry','layer','waferu','waferv'],inplace=True)
df_econ['simenergy'] = df_simtotal
df_econ.drop(columns='subentry',inplace=True)

#filter out zero simenergy
df_econ_wsimenergy = df_econ[df_econ.simenergy > 0]
df_econ_wsimenergy['layer'] = df_econ_wsimenergy.index.get_level_values('layer') 


b, id_list = prepare_data(df_econ_wsimenergy, df_econ)
b.reset_index(inplace=True)
b.set_index(['entry'],inplace=True)
ref = b.drop(columns=['level_0']).loc[0]
ref['data'] = 0
ref['simenergy'] = 0

entries = []

for i in range(1000):
    copy_ref = ref.copy().reset_index()
    copy_b = b.loc[i].reset_index()
    copy_ref['entry'] = i
    copy_ref.set_index(['layer', 'waferu', 'waferv', 'id'])
    copy_b.set_index(['layer', 'waferu', 'waferv', 'id'])
    copy_ref.loc[copy_b.index] =  copy_b.loc[copy_b.index]
    copy_ref.reset_index()
    entries.append(copy_ref.set_index(['entry']))
    
final = pd.concat(entries)

final = final.drop(columns=['id'])

final.to_hdf('data.h5', key='df', mode='w')  
