from __future__ import print_function
import keras
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
import numpy as np
np.random.seed(10000)
import tensorflow as tf
from tensorflow.keras import layers, Model
def model_user(input_shape,labels_dim):
    inputs=Input(shape=input_shape, dtype='float64')
    middle_layer=Dense(1024,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs)
    middle_layer=Dense(512,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    outputs_logits=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(middle_layer)
    outputs=Activation('softmax', dtype='float64')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    return model



def model_user_3layer(input_shape,labels_dim):
    inputs=Input(shape=input_shape, dtype='float64')
    middle_layer=Dense(512,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs)
    middle_layer=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    outputs_logits=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(middle_layer)
    outputs=Activation('softmax', dtype='float64')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_user_logits(input_shape,labels_dim):
    inputs=Input(shape=input_shape)
    middle_layer=Dense(1024,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs)
    middle_layer=Dense(512,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    outputs_logits=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(middle_layer)
    model = Model(inputs=inputs, outputs=outputs_logits)
    return model
def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model



def model_defense_logits(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(x_b)
    # outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs_pre)
    return model

def model_defense_optimize(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Activation('softmax')(inputs_b)
    x_b=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


def model_attack_nn(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(512,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs_b)
    x_b=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

def model_attack_nn_2layer(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

def model_attack_nn_4layer(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b = Dense(1024, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100), activation='relu')(inputs_b)
    x_b=Dense(512,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(256,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

def model_attack_nsh(input_shape_f_x,input_shape_y,labels_dim):

    inputs_f_x = layers.Input(shape=input_shape_f_x)
    x1 = layers.Dense(1024, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(inputs_f_x)
    x1 = layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(x1)
    x1 = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(x1)

    inputs_y = layers.Input(shape=input_shape_y)
    x2 = layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(inputs_y)
    x2 = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(x2)


    concatenated = layers.concatenate([x1, x2])
    x3 = layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(concatenated)
    x3 = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(x3)
    outputs_pre = layers.Dense(labels_dim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(x3)
    outputs = layers.Activation('sigmoid')(outputs_pre)


    model = Model(inputs=[inputs_f_x, inputs_y], outputs=outputs)
    return model

