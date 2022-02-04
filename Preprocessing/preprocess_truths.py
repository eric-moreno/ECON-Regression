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

df_gen_individual = []

for i in range(len(files[:48])): 
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

df_gen.to_hdf('truths_0-48k.h5', key='df', mode='w')         
