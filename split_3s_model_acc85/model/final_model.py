# -*- coding: utf-8 -*-
# @Time    : 2019/2/8 16:38
# @Author  : shaoeric
# @Email   : shaoeric@foxmail.com

from keras.layers import Input, BatchNormalization, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GaussianNoise, Reshape, CuDNNGRU, Add
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from os import environ
from keras import regularizers, optimizers
from split_3s_model_acc85.model.utils import *
from split_3s_model_acc85.model.Attention import Attention

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ["CUDA_VISIBLE_DEVICES"] = "0"

# [1.5907280000139412, 0.8518105849582173] loss acc

mfcc = np.load('../features/pre_process_mfcc.npy')
logfbank = np.load('../features/concat_logbank.npy')
y = np.load('../features/target.npy')
y = to_categorical(y, 10)

mfcc_train, mfcc_test, logfbank_train, logfbank_test, y_train, y_test = train_test_split(mfcc,logfbank, y, random_state=43, test_size=0.2)

# mfcc input
def get_mfcc_model():
    input_layer = Input(shape=(1, 130, 40), name='mfcc_input')
    x = GaussianNoise(0.08)(input_layer)
    x = Conv2D(data_format='channels_first', filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=30, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(0.01), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(0.01), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=2, padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=30, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(0.01), data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(0.01), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=2, padding='same', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(10, activation='softmax', name='mfcc_output', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.2))(x)
    model = Model(input=input_layer, output=x)
    # model.compile(optimizer=optimizers.sgd(lr=1e-4, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_logfbank_model():
    regularize = 0.01
    input_layer = Input(shape=(1, 120, 120), name='attention_logbank_input')
    x = GaussianNoise(0.03)(input_layer)
    x = Conv2D(data_format='channels_first', filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(regularize))(x)
    x = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal',
               kernel_regularizer=regularizers.l2(regularize), data_format='channels_first')(x)
    x = MaxPooling2D(pool_size=(1, 3), padding='same', data_format='channels_first')(x)
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
    x = Reshape(target_shape=(32, 25))(x)
    x = Dropout(0.3)(x)
    x = CuDNNGRU(34, return_sequences=True)(x)
    x = Attention(x.shape[1], name='attention')(x)
    x = Dropout(0.3)(x)
    x = Dense(10, activation='softmax', name='attention_logbank_output', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(regularize))(x)
    model = Model(input=input_layer, output=x)
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
model.compile(optimizer=optimizers.adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x={'mfcc_input': np.expand_dims(mfcc_train, axis=1),'attention_logbank_input': np.expand_dims(logfbank_train, axis=1)}, y=y_train, batch_size=300, epochs=50, validation_split=0.2)


print(model.evaluate(x={'mfcc_input': np.expand_dims(mfcc_test, axis=1),'attention_logbank_input': np.expand_dims(logfbank_test, axis=1)}, y=y_test))
