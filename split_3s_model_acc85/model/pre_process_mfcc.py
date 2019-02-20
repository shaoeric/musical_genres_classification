# -*- coding: utf-8 -*-
# @Time    : 2019/2/8 15:00
# @Author  : shaoeric
# @Email   : shaoeric@foxmail.com

from keras.layers import Input, BatchNormalization, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GaussianNoise
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from os import environ
from keras import regularizers, optimizers
from split_3s_model_acc85.model.utils import *

"""
# [1.2020697935045928, 0.8217270195318132] loss acc
"""
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ["CUDA_VISIBLE_DEVICES"] = "0"

X = np.load('../features/pre_process_mfcc.npy')
y = np.load('../features/target.npy')
y = to_categorical(y, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=0.2)


# mfcc input
def get_model():
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
    print(model.summary())
    return model


def train_only_mfcc(model, weights, epoch, X_train_argued, y_train_argued):
    # 过拟合严重
    history = model.fit({'mfcc_input': X_train_argued}, {'mfcc_output': y_train_argued}, epochs=epoch, validation_split=0.2, batch_size=300)
    model.save_weights(weights)
    return history


def train(model, weights, epoch):
    # 生成训练数据
    X_train_argued, y_train_argued = argument_data(X_train,y_train, 300, 400, 130, 40)
    X_train_argued, y_train_argued = get_argument_data_2d(X_train_argued, y_train_argued)
    history = train_only_mfcc(model, weights, epoch, X_train_argued, y_train_argued)
    plot_curve(history)


model = get_model()
model.compile(optimizer=optimizers.adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# train(model, '../weights/pre_process_mfcc.hdf5', epoch=50)

model.load_weights('../weights/pre_process_mfcc.hdf5')
print(model.evaluate(np.expand_dims(X_test,axis=1), y_test))


