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


files = os.listdir('/eos/uscms/store/user/cmantill/HGCAL/EleGun/Jan24/EGM-Phase2HLTTDRWinter20GS-00012-NoPU/crab_gun_electrons_01_25_22/220126_181410/0000/')

df_tc_individual = []
df_econ_individual = []

files = ['ntuple_46.root',
'ntuple_47.root',
'ntuple_48.root',
'ntuple_5.root',
'ntuple_6.root',
'ntuple_7.root',
'ntuple_8.root',
'ntuple_9.root']

for i in range(len(files[:8])): 
    file = files[i]
    print(file)    
    fname = "/eos/uscms/store/user/cmantill/HGCAL/EleGun/Jan24/EGM-Phase2HLTTDRWinter20GS-00012-NoPU/crab_gun_electrons_01_25_22/220126_181410/0000/%s"%(file) 
    
    with uproot.open(fname) as f:
        ev_dict = f["FloatingpointAutoEncoderStrideDummyHistomaxGenmatchGenclustersntuple/HGCalTriggerNtuple"]
    
        arrays_toread = [
        "econ_index","econ_data",
        #"econ_subdet","econ_zside",
        "econ_layer","econ_waferu","econ_waferv","econ_wafertype",
        "tc_simenergy",
        #"tc_subdet","tc_zside",
        "tc_layer","tc_waferu","tc_waferv","tc_wafertype",
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


        temp_tc = ak.to_pandas(tc).reset_index()
        del tc
        temp_econ = ak.to_pandas(econ).reset_index()
        del econ

        i += 40
        temp_tc['entry'] = (temp_tc['entry'].to_numpy()+ i*1000)
        temp_econ['entry'] = (temp_econ['entry'].to_numpy()+ i*1000)

        df_tc_individual.append(temp_tc)
        df_econ_individual.append(temp_econ)

        
df_tc = pd.concat(df_tc_individual)
del df_tc_individual
df_econ = pd.concat(df_econ_individual)
del df_econ_individual

print("done loading")

df_simtotal = df_tc.groupby(['entry','layer','waferu','waferv'])["simenergy"].sum()
del df_tc

#Prepare df_econ
df_econ.reset_index(inplace=True)
df_econ.set_index(['entry','layer','waferu','waferv'],inplace=True)
df_econ['simenergy'] = df_simtotal
df_econ.drop(columns='subentry',inplace=True)

#filter out zero simenergy
df_econ_wsimenergy = df_econ[df_econ.simenergy > 0]
df_econ_wsimenergy['layer'] = df_econ_wsimenergy.index.get_level_values('layer') 


# select wafers from df_econ in top n(=50) wafer simenergies 
id_list = np.load('id_list.npy') 

b = df_econ[df_econ['id'].isin(id_list)]
print(b)
print(b.shape)

# create temporary template array that contains all wafers 
b.reset_index(inplace=True)
b.set_index(['entry'],inplace=True)
#temp = b.drop(columns=['level_0']).loc[0]

temp = pd.read_hdf('temp_shell.h5')
print(temp)
print(temp.shape)

# clear template array of data
#temp['data'] = 0
#temp['simenergy'] = 0
#temp.to_hdf('temp_shell.h5', key='df', mode='w') 

# fill templates with available wafer data
entries = []
for i in range(8000):
    i += 40000
    copy_temp = temp.copy().reset_index()
    copy_b = b.loc[i].reset_index()
    copy_temp['entry'] = i
    copy_temp.set_index(['layer', 'waferu', 'waferv', 'id'])
    copy_b.set_index(['layer', 'waferu', 'waferv', 'id'])
    copy_temp.loc[copy_b.index] =  copy_b.loc[copy_b.index]
    copy_temp.reset_index()
    entries.append(copy_temp.set_index(['entry']))

# create final arrays
final = pd.concat(entries)
del entries

final = final.drop(columns=['id'])

final.to_hdf('train_40-48k.h5', key='df', mode='w')  