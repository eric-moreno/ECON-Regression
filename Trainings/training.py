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
from model import model_ConvDNN, model_DNN, model_DNN_flatten, model_1dConvDNN
from keras import regularizers, Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import mean_absolute_error, MeanAbsoluteError, mean_squared_error, MeanSquaredError

train_list = [pd.read_hdf('../Preprocessing/train_0-20k_chargetotal.h5'), pd.read_hdf('../Preprocessing/train_20-40k_chargetotal.h5')]
train_list = pd.concat(train_list) 
#train_list.drop(columns=['index'], inplace=True)
train_data = train_list.to_numpy()
train_data = train_data.reshape((40000, 800, 5))[:, :, -2:].reshape((40000, 50, int(16*2))) #DNN 

#train_data_wafer = train_data.reshape((40000, 50, 16, 5))[:, :, :, -2].reshape((40000, 50, 16))
#train_data_location = train_data.reshape((40000, 50, 16, 5))[:, :, 0, :3].reshape((40000, 50, 3))
#train_data_charge = train_data.reshape((40000, 50, 16, 5))[:, :, 0, -1].reshape((40000, 50, 1))
#train_data = np.concatenate([train_data_charge, train_data_wafer], axis=2)

truth = pd.read_hdf('../Preprocessing/truths_0-48k.h5')
truth = truth.to_numpy()[:40000, 5]

model = model_DNN(train_data) 
model.compile(optimizer='adam', loss='mse')
model.summary()
outdir = 'DNN_v1_totalcharge_pTreg'
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

print(model.predict(train_data))
print(truth)
