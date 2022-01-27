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


train_data = pd.read_hdf('../Preprocessing/data.h5').to_numpy().reshape((1000, 50, 16, 6))
truth = pd.read_hdf('../Preprocessing/truths.h5').to_numpy()[:,3]

scaler = StandardScaler()
truth = scaler.fit_transform(truth.reshape(-1, 1))

#print(train_data.to_numpy().reshape((1000, 800, 6)))
#print(truth.to_numpy()[:,3])

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


model = model_ConvDNN(train_data) 
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
outdir = '.'
model_type = "CNN" 
# fit the model to the data
nb_epochs = 300
batch_size = 256
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('%s/best_model_%s.hdf5'%(outdir, model_type), save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(train_data, truth, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
model.save('%s/last_model.hdf5'%(outdir))
