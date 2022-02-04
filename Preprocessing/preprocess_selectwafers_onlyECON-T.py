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

files = ['ntuple_1.root', 'ntuple_10.root', 'ntuple_11.root', 'ntuple_12.root', 'ntuple_13.root', 'ntuple_14.root', 'ntuple_15.root', 'ntuple_16.root', 'ntuple_17.root', 'ntuple_18.root', 'ntuple_19.root', 'ntuple_2.root', 'ntuple_20.root', 'ntuple_21.root', 'ntuple_22.root', 'ntuple_23.root', 'ntuple_24.root', 'ntuple_25.root', 'ntuple_26.root', 'ntuple_27.root', 'ntuple_28.root', 'ntuple_29.root', 'ntuple_3.root', 'ntuple_30.root', 'ntuple_31.root', 'ntuple_32.root', 'ntuple_33.root', 'ntuple_34.root', 'ntuple_35.root', 'ntuple_36.root', 'ntuple_37.root', 'ntuple_38.root', 'ntuple_39.root', 'ntuple_4.root', 'ntuple_40.root', 'ntuple_41.root', 'ntuple_42.root', 'ntuple_43.root', 'ntuple_44.root', 'ntuple_45.root', 'ntuple_46.root', 'ntuple_47.root', 'ntuple_48.root', 'ntuple_5.root', 'ntuple_6.root', 'ntuple_7.root', 'ntuple_8.root', 'ntuple_9.root']

df_tc_individual = []
df_econ_individual = []

# test files = ['ntuple_46.root','ntuple_47.root','ntuple_48.root','ntuple_5.root','ntuple_6.root','ntuple_7.root','ntuple_8.root', 'ntuple_9.root']

for i in range(48): 
    file = files[i]  
    print(file)
    fname = "/eos/uscms/store/user/cmantill/HGCAL/EleGun/Jan24/EGM-Phase2HLTTDRWinter20GS-00012-NoPU/crab_gun_electrons_01_25_22/220126_181410/0000/%s"%(file) 
    
    with uproot.open(fname) as f:
        ev_dict = f["FloatingpointAutoEncoderStrideDummyHistomaxGenmatchGenclustersntuple/HGCalTriggerNtuple"]
    
        arrays_toread = [
        "econ_index","econ_data",
        "econ_id"
        ]
        events = ev_dict.arrays(arrays_toread)

        #Separate the data sets
        econ = ak.zip({
            "index": events['econ_index'],
            "id":events['econ_id'],
            "data": events["econ_data"],
        })

        temp_econ = ak.to_pandas(econ).reset_index()
        del econ
       
        temp_econ['entry'] = (temp_econ['entry'].to_numpy()+ i*1000)

        df_econ_individual.append(temp_econ)

df_econ = pd.concat(df_econ_individual)
del df_econ_individual

print("done loading")
df_econ.set_index(['entry'],inplace=True)
df_econ.drop(columns=['subentry'], inplace=True)

id_list = np.load('id_list.npy')
b = df_econ[df_econ['id'].isin(id_list)]

temp = b.loc[0]
temp['data'] = 0

print(temp)

# select wafers from df_econ in top n(=50) wafer simenergies 

# fill templates with available wafer data
entries = []
for i in range(48000):
    copy_temp = temp.copy().reset_index()
    copy_b = b.loc[i].reset_index()
    copy_temp['entry'] = i
    copy_temp.set_index(['id', 'index'])
    copy_b.set_index(['id', 'index'])
    copy_temp.loc[copy_b.index] =  copy_b.loc[copy_b.index]
    copy_temp.reset_index()
    entries.append(copy_temp.set_index(['entry']))

# create final arrays
final = pd.concat(entries)

print(final.loc[0])
del entries

final = final.drop(columns=['id'])

final.to_hdf('data_0-48k.h5', key='df', mode='w')  