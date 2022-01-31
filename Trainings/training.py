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
#tf.logging.set_verbosity(tf.logging.ERROR) 

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, MaxPooling2D, UpSampling1D, BatchNormalization, Flatten, Reshape, GRU, Conv2D, Bidirectional, Concatenate
from keras.models import Model
from keras import regularizers, Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import mean_absolute_error, MeanAbsoluteError, mean_squared_error, MeanSquaredError

train_list = [pd.read_hdf('../Preprocessing/train_0-20k.h5'), pd.read_hdf('../Preprocessing/train_20-40k.h5')]
train_data = pd.concat(train_list).to_numpy().reshape((40000, 800, 6))
truth = pd.read_hdf('../Preprocessing/truths.h5').to_numpy()[:40000,3]
print(train_data.shape)

print(train_data[:,:, -1].sum(axis=1)/16) 
print(truth) 

mse = MeanSquaredError(reduction='none')
mse_distrib = mse(truth.reshape(-1, 1), train_data[:,:, -1].sum(axis=1).reshape(-1, 1)/16).numpy()
print(mse_distrib)
plt.title('Loss distribution Sum(Simenergy)')
plt.xlabel('Count')
plt.ylabel('Loss')
plt.figure()
n, bins, patches = plt.hist(mse_distrib, 10, facecolor='blue', alpha=0.5, range=(0, 2000))
plt.savefig('mse_distrib_sumsimenergy.jpg')

#scaler = StandardScaler()
#truth = scaler.fit_transform(truth.reshape(-1, 1))

#print(train_data.to_numpy().reshape((1000, 800, 6)))
#prerint(truth.to_numpy()[:,3])

def model_ConvDNN(X):
    inputs = Input(shape=(X.shape[1],X.shape[2], X.shape[3]))
    L1 = Conv2D(32, (2, 2), activation="relu")(inputs)
    L2 = MaxPooling2D((2,2))(L1)
    print(L2.shape)
    L3 = Conv2D(16, (2, 2), activation="relu")(L2)
    print(L3.shape)
    L4 = MaxPooling2D((2,2))(L3)
    print(L4.shape)
    L5 = Conv2D(8, (2, 2), activation="relu")(L4)
    L6 = MaxPooling2D((2,2))(L5)
    x = Flatten()(L6)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs = output)
    return model  

def model_DNN(X): 
    inputs = Input(shape=(X.shape[1],X.shape[2]))
    x = Dense(1, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


model = model_DNN(train_data) 
model.compile(optimizer='adam', loss='mse')
model.summary()
outdir = '.'
model_type = "DNN" 
# fit the model to the data
nb_epochs = 100
batch_size = 16
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('%s/best_model_%s.hdf5'%(outdir, model_type), save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(train_data, truth, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
model.save('%s/last_model.hdf5'%(outdir))

fig = plt.figure()
plt.plot(range(nb_epochs),history['loss'],label='training loss', color = "blue")
plt.plot(range(nb_epochs),history['val_loss'],label='validation loss', color = "red")
plt.xlabel('epochs')
plt.ylabel('MSE Loss')
plt.title("Training and validation loss")
fig.savefig('loss.jpg')

