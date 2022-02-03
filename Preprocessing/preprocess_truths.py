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
files.sort()

df_gen_individual = []

for i in range(len(files[:49])): 
    file = files[i]
    print(file)
    
    fname = "/eos/uscms/store/user/cmantill/HGCAL/EleGun/Jan24/EGM-Phase2HLTTDRWinter20GS-00012-NoPU/crab_gun_electrons_01_25_22/220126_181410/0000/%s"%(file)
    ev_dict = uproot.open(fname)["FloatingpointAutoEncoderStrideDummyHistomaxGenmatchGenclustersntuple/HGCalTriggerNtuple"]
    
    arrays_toread = [
    "gen_pt","gen_energy","gen_eta","gen_phi",
    "genpart_pt","genpart_energy",
    ]
    
    events = ev_dict.arrays(arrays_toread)

    gen = ak.zip({
        "pt": events["gen_pt"],
        "energy": events["gen_energy"],
        "eta": events["gen_eta"],
        "phi": events["gen_phi"],
    })
    
    temp_gen = ak.to_pandas(gen).reset_index()
    print(temp_gen.shape)
    df_gen_individual.append(temp_gen)

df_gen = pd.concat(df_gen_individual)
print(df_gen)
print(df_gen.shape)
print(df_gen.to_numpy()[:,0])
del df_gen_individual

df_gen.to_hdf('truths.h5', key='df', mode='w')         
