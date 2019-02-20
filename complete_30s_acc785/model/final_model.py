# -*- coding: utf-8 -*-
# @Time    : 2019/2/8 16:38
# @Author  : shaoeric
# @Email   : shaoeric@foxmail.com

from keras.layers import Input, BatchNormalization, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GaussianNoise, Reshape, CuDNNGRU, Concatenate, Add, Dot
from keras.models import Model, Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from os import environ
from keras import regularizers, optimizers
from complete_30s_acc785.model.utils import *
from complete_30s_acc785.model.Attention import Attention

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ["CUDA_VISIBLE_DEVICES"] = "0"

# [2.5030006694793703, 0.785] loss acc

mfcc = np.load('../features/pre_process_mfcc.npy')
logfbank = np.load('../features/concat_logbank.npy')
y = np.load('../features/target.npy')
y = to_categorical(y, 10)

mfcc_train, mfcc_test, logfbank_train, logfbank_test, y_train, y_test = train_test_split(mfcc,logfbank, y, random_state=43, test_size=0.2)

noise = 0.22

def get_mfcc_model():
    input_layer = Input(shape=(1, 1290, 40), name='mfcc_input')
    x = GaussianNoise(noise)(input_layer)
    x = Conv2D(data_format='channels_first', filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=30, kernel_size=3, padding='same', activation='relu',kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01),data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=2, padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=30, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(0.01), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=2, padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Flatten(name='mfcc_flatten')(x)
    x = Dropout(0.3)(x)
    x = Dense(10, activation='softmax', name='mfcc_output', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.2))(x)
    model = Model(inputs=input_layer, outputs=x)

    return model

def get_logfbank_model():
    regularize = 0.01
    input_layer = Input(shape=(1, 1197, 120), name='attention_logbank_input')
    x = GaussianNoise(noise)(input_layer)
    x = Conv2D(data_format='channels_first', filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(regularize))(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(regularize), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(regularize), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=2, padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(regularize), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=2, padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(regularize), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=2, padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Reshape(target_shape=(17, 32 * 15))(x)
    x = Dropout(0.3)(x)
    x = CuDNNGRU(34, return_sequences=True)(x)
    x = Attention(x.shape[1], name='attention')(x)
    x = Dropout(0.3)(x)
    x = Dense(10, activation='softmax', name='attention_logbank_output', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(regularize))(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


m1 = get_mfcc_model()
m2 = get_logfbank_model()
m1.load_weights('../weights/pre_process_mfcc.hdf5')
m2.load_weights('../weights/attention_logbank.hdf5')

for layer in m1.layers:
    layer.trainable = False

for layer in m2.layers:
    layer.trainable = False


out = Add()([m1.output, m2.output])

model = Model(inputs=[m1.input, m2.input], outputs=out)
print(model.summary())
model.compile(optimizer=optimizers.adam(lr=1e-4, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x={'mfcc_input': np.expand_dims(mfcc_train, axis=1),'attention_logbank_input': np.expand_dims(logfbank_train, axis=1)}, y=y_train, batch_size=110, epochs=100, validation_split=0.2)

# model.save('../weights/final.hdf5')
print(model.evaluate(x={'mfcc_input': np.expand_dims(mfcc_test, axis=1),'attention_logbank_input': np.expand_dims(logfbank_test, axis=1)}, y=y_test))
