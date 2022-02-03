import numpy as np
import keras 
import tensorflow as tf
from keras.layers import Input, Dense, MaxPooling2D, BatchNormalization, Flatten, Reshape, Conv2D
from keras.models import Model
from keras import regularizers, Sequential 

def model_ConvDNN(X):
    inputs = Input(shape=(X.shape[1], X.shape[2],X.shape[3]))
    L1 = Conv2D(4, (2, 2), activation="relu")(inputs)
    L2 = MaxPooling2D((2,2))(L1)
    L3 = Conv2D(8, (2, 2), activation="relu")(L2)
    L4 = MaxPooling2D((2,2))(L3)
    L5 = Conv2D(16, (2, 2), activation="relu")(L4)
    L6 = MaxPooling2D((2,2))(L5)
    x = Flatten()(L6)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs = output)
    return model  

def model_DNN(X): 
    inputs = Input(shape=(X.shape[1],X.shape[2]))
    x = Dense(2, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


