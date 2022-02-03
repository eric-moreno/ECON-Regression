import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import setGPU
import keras 
from numpy.random import seed
import tensorflow as tf
from model import model_ConvDNN, model_DNN
from keras import regularizers, Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import mean_absolute_error, MeanAbsoluteError, mean_squared_error, MeanSquaredError

train_list = [pd.read_hdf('../Preprocessing/train_0-20k.h5'), pd.read_hdf('../Preprocessing/train_20-40k.h5')]
train_data = pd.concat(train_list).to_numpy().reshape((40000, 800, 6))#[:, :, :, -2:]
truth = pd.read_hdf('../Preprocessing/truths.h5')
truth = truth.to_numpy()[:40000, 3]
print(truth) 

#scaler = StandardScaler()
#truth = scaler.fit_transform(truth.reshape(-1, 1))

#print(train_data.to_numpy().reshape((1000, 800, 6)))
#prerint(truth.to_numpy()[:,3])


model = model_DNN(train_data) 
model.compile(optimizer='adam', loss='mse')
model.summary()
outdir = 'DNN_v1'
model_type = "DNN" 
# fit the model to the data
nb_epochs = 100
batch_size = 16
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('%s/best_model_%s'%(outdir, model_type), save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(train_data, truth, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
model.save('%s/last_model_%s'%(outdir, model_type))

fig = plt.figure()
plt.plot(history['loss'][1:],label='training loss', color = "blue")
plt.plot(history['val_loss'][1:],label='validation loss', color = "red")
plt.xlabel('epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title("Training and validation loss")
fig.savefig('loss.jpg')
