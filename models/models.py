from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json
import numpy as np
import random
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.layers import BatchNormalization, LSTM, Dropout, Dense, TimeDistributed, Masking, Activation, Input, Reshape, Embedding, Bidirectional
from keras import optimizers,regularizers
from keras.callbacks import ModelCheckpoint

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def get_optimizer(lr=0.0005):
    adam = optimizers.Adam(lr=lr)
    return adam

def network_mortality(input_size, catg_len=429, embedding_dim=5, numerical=True, categorical=True, ohe=False):
    if numerical and categorical:
        if ohe:
            input1 = Input(shape=(input_size, 13))
            input2 = Input(shape=(input_size, 429))
            inp = keras.layers.Concatenate(axis=-1)([input1, input2])
        else:
            input1 = Input(shape=(input_size, 13))
            input2 = Input(shape=(input_size, 7))
            x2 = Embedding(catg_len, embedding_dim)(input2)
            x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)
            inp = keras.layers.Concatenate(axis=-1)([input1, x2])
    elif numerical:
        input1 = Input(shape=(input_size, 13))
        inp = input1 
    elif categorical:
        if ohe:
            input1 = Input(shape=(input_size, 7))
            inp = Reshape((int(input1.shape[1]),int(input1.shape[2]*input1.shape[3])))(input1)
        else:
            input1 = Input(shape=(input_size, 7))
            x1 = Embedding(catg_len, embedding_dim)(input1)
            inp = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)

    mask = Masking(mask_value=0.,name="maski")(inp)

    lstm1 = Bidirectional(LSTM(units=64,kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal' ,name= "1stl",return_sequences=True))(mask)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.6)(lstm1)

    lstm2 = Bidirectional(LSTM(units=32,kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal' ,name= "2ndl",return_sequences=True))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(0.4)(lstm2)

    lstm3 = Bidirectional(LSTM(units=32,kernel_initializer='glorot_normal' ,name= "3rdl",return_sequences=False))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    lstm3 = Dropout(0.4)(lstm3)

    out = Dense(1,activation="sigmoid")(lstm3)

    if numerical and categorical:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)

    adam = get_optimizer(lr=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=adam ,metrics=[f1,sensitivity, specificity,'accuracy'])
    
    return model


def network_los(input_size, catg_len=429, embedding_dim=5, numerical=True, categorical=True):
    if numerical and categorical:
          if ohe:
            input1 = Input(shape=(input_size, 13))
            input2 = Input(shape=(input_size, 429))
            inp = keras.layers.Concatenate(axis=-1)([input1, input2])
        else:
            input1 = Input(shape=(input_size, 13))
            input2 = Input(shape=(input_size, 7))
            x2 = Embedding(catg_len, embedding_dim)(input2)
            x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)
            inp = keras.layers.Concatenate(axis=-1)([input1, x2])

        # input1 = Input(shape=(input_size, 13))
        # input2 = Input(shape=(input_size, 7))
        # x2 = Embedding(catg_len, embedding_dim)(input2)
        # x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)
        # inp = keras.layers.Concatenate(axis=-1)([input1, x2])
    
    elif numerical:
        input1 = Input(shape=(input_size, 13))
        inp = input1
    
    elif categorical:
        if ohe:
            input1 = Input(shape=(input_size, 429))
            inp = Reshape((int(input1.shape[1]),int(input1.shape[2]*input1.shape[3])))(input1)
        else:
            input1 = Input(shape=(input_size, 7))
            x1 = Embedding(catg_len, embedding_dim)(input1)
            inp = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)

    mask = Masking(mask_value=0.,name="maski")(inp)
    lstm1 = Bidirectional(LSTM(units=128, name= "lstm1",kernel_initializer='glorot_normal',return_sequences=True))(mask) 
    lstm1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(units=128, name= "lstm2",kernel_initializer='glorot_normal',return_sequences=True)(lstm1) 
    lstm2 = Dropout(0.2)(lstm2)
    lstm3 = LSTM(units=128, name= "lstm3",kernel_initializer='glorot_normal',return_sequences=True)(lstm2) 
    lstm3 = Dropout(0.2)(lstm3)
    out = TimeDistributed(Dense(1,activation="relu"))(lstm3)

    if numerical and categorical:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)   
    adam = get_optimizer(lr=0.005)
    
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
    return model 

def network_phenotyping(input_size, catg_len=429, embedding_dim=5, numerical=True, categorical=True):
    if numerical and categorical:
        input1 = Input(shape=(input_size, 13))

        input2 = Input(shape=(input_size, 7))
        x2 = Embedding(catg_len, embedding_dim)(input2)
        x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)

        inp = keras.layers.Concatenate(axis=-1)([input1, x2])
    
    elif numerical:
        input1 = Input(shape=(input_size, 13))
        inp = input1
    
    elif categorical:
        input1 = Input(shape=(input_size, 7))
        x1 = Embedding(catg_len, embedding_dim)(input1)
        inp = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)

    mask = Masking(mask_value=0.,name="maski")(inp)

    lstm1 = Bidirectional(LSTM(units=50,kernel_initializer='glorot_normal' ,name= "1stl",return_sequences=True))(mask)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.2)(lstm1)

    lstm2 = Bidirectional(LSTM(units=50,kernel_initializer='glorot_normal' ,name= "3rdl",return_sequences=False))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(0.2)(lstm2)

    out = Dense(25,activation="sigmoid")(lstm2)

    if numerical and categorical:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)
    
    adam = get_optimizer(lr=0.001)
    model.compile(loss="binary_crossentropy" ,optimizer=adam, metrics=[f1,'accuracy'])
    
    return model

def network_decompensation(input_size, catg_len=429, embedding_dim=5, numerical=True, categorical=True):
    if numerical and categorical:
        input1 = Input(shape=(input_size, 13))

        input2 = Input(shape=(input_size, 7))
        x2 = Embedding(catg_len, embedding_dim)(input2)
        x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)

        inp = keras.layers.Concatenate(axis=-1)([input1, x2])
    
    elif numerical:
        input1 = Input(shape=(input_size, 13))
        inp = input1
    
    elif categorical:
        input1 = Input(shape=(input_size, 7))
        x1 = Embedding(catg_len, embedding_dim)(input1)
        inp = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)

    mask = Masking(mask_value=0.,name="maski")(inp)

    lstm1 = Bidirectional(LSTM(units=128, name= "lstm1",kernel_initializer='glorot_normal',return_sequences=True))(mask) 
    lstm1 = Dropout(0.3)(lstm1)

    lstm2 = LSTM(units=8, name= "lstm2",kernel_initializer='glorot_normal',return_sequences=True)(lstm1) 
    lstm2 = Dropout(0.2)(lstm2)

    lstm3 = LSTM(units=8, name= "lstm3",kernel_initializer='glorot_normal',return_sequences=True)(lstm2) 
    lstm3 = Dropout(0.2)(lstm3)

    out = TimeDistributed(Dense(1,activation="sigmoid"))(lstm3)
    
    if numerical and categorical:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)

    adam = get_optimizer(lr=0.005)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=[f1,'accuracy'])
    return model