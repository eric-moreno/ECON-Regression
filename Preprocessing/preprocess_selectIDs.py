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


files = os.listdir('/eos/uscms/store/user/cmantill/HGCAL/EleGun/Jan24/EGM-Phase2HLTTDRWinter20GS-00012-NoPU/crab_gun_electrons_01_25_22/220126_181410/0000/')

df_tc_individual = []
df_econ_individual = []

for i in range(len(files[:20])): 
 
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
df_econ['simenergy'] = df_simtotal #heavy RAM step
df_econ.drop(columns='subentry',inplace=True)

#filter out zero simenergy
df_econ_wsimenergy = df_econ[df_econ.simenergy > 0]
df_econ_wsimenergy['layer'] = df_econ_wsimenergy.index.get_level_values('layer') 

# select wafers from df_econ in top n(=50) wafer simenergies 
b, id_list = prepare_data(df_econ_wsimenergy, df_econ)
print(id_list) 
np.save('id_list.npy', id_list)

#For reference: 20k id_list = [2989252864, 2989777152, 2990301440, 2987155712, 2988728576, 2988204288, 2987680000, 2986631424, 2990825728, 2991350016, 2991874304, 2992398592, 2992922880, 2989253120, 2989777408, 2988728832, 2989785600, 2990309888, 2990301696, 2990834176, 2988204544, 2990825984, 2991358464, 2989261312, 2993447168, 2991882752, 2991350272, 2988737024, 2992407040, 2991874560, 2987680256, 2992931328, 2988212736, 2992398848, 2993455616, 3020185856, 2992923136, 2989768960, 2987155968, 2989244672, 2990293248, 2990833920, 2990309632, 2991358208, 2993447424, 2988720384, 2989785344, 2987688448, 2990817536, 2991882496]


