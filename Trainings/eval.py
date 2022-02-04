import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import keras 
#import setGPU
from numpy.random import seed
import tensorflow as tf
from model import model_ConvDNN, model_DNN
from keras import regularizers, Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import mean_absolute_error, MeanAbsoluteError, mean_squared_error, MeanSquaredError

model_outdir = 'DNN_v1_onlywafers'
model_name = 'best_model_DNN'
model_type = 'DNN' 

#Load train data
train_list = pd.read_hdf('../Preprocessing/data_0-48k.h5')
train_list.drop(columns=['index'], inplace=True)
train_data = train_list.to_numpy().reshape((48000, 50, 16))[:40000]
test_data = train_list.to_numpy().reshape((48000, 50, 16))[40000:]
truth = pd.read_hdf('../Preprocessing/truths_0-48k.h5')
train_truth = truth.to_numpy()[:40000, 3]
test_truth = truth.to_numpy()[40000:, 3]

#Predict train data
model = keras.models.load_model('%s/%s'%(model_outdir,model_name))
pred = model.predict(train_data)
np.save('%s/prediction_training_0-40k.npy'%(model_outdir), pred) 

#Predict test data
pred = model.predict(test_data) 
np.save('%s/prediction_test_40-48k.npy'%(model_outdir), pred) 

mse = MeanSquaredError(reduction='none')

'''
#Loss distribution for naive simenergy sum calculation
mse_distrib = mse(test_truth.reshape(-1, 1), test_data.reshape(8000, 800, 6)[:,:, -1].sum(axis=1).reshape(-1, 1)/16).numpy()
plt.figure()
plt.title('Test Loss Distribution')
plt.ylabel('Count')
plt.xlabel('MSE Loss')
n, bins, patches = plt.hist(mse_distrib, 30, facecolor='blue', label='Sum(SimEnergy)', alpha=0.5, range=(0, 6000))
'''

#Loss distribution from model 
mse_distrib = mse(test_truth.reshape(-1, 1), pred.reshape(-1,1)).numpy()
n, bins, patches = plt.hist(mse_distrib, 30, facecolor='red',label='%s model'%(model_type), alpha=0.5, range=(0, 60000))
plt.legend()
plt.savefig('%s/mse_distrib_sumsimenergy.jpg'%(model_outdir))


