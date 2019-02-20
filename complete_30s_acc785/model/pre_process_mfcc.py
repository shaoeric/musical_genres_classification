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
from complete_30s_acc785.model.utils import *

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
    mfcc.npy 是提取的原始mfcc文件，未标准化。mfcc_normed_axis0是0-1标准化后的特征文件
    mfcc_not_normed.hdf5是未标准化训练的模型参数，mfcc.hdf5是标准化训练的模型参数
    [2.515409245491028, 0.635] 未标准化的mfcc 2d卷积分数
    [2.190826025009155, 0.695]  是标准化的mfcc 2d卷积分数
    用1d卷积+时序模型效果不好，收敛慢，分数也不理想
    [2.0766567850112914, 0.575] 未标准化的mfcc concat 2d卷积分数(mfcc,mfcc_d,mfcc_dd)
    [2.5405269050598145, 0.525] 标准化的mfcc concat 2d卷积
    [2.100687427520752, 0.635] 未标准化的mfcc stack 2d卷积 (mfcc;mfcc_d;mfcc_dd)
    # [1.7943829917907714, 0.75] 未标准化的自定义预处理pre_process_mfcc.hdf5
"""

X = np.load('../features/pre_process_mfcc.npy')
y = np.load('../features/target.npy')
y = to_categorical(y, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=0.2)


# mfcc input
def get_model():
    input_layer = Input(shape=(1, 1290, 40), name='mfcc_input')
    x = GaussianNoise(0.01)(input_layer)
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
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(10, activation='softmax', name='mfcc_output', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.2))(x)
    model = Model(input=input_layer, output=x)
    print(model.summary())
    return model


def train_only_mfcc(model, weights, epoch, X_train_argued, y_train_argued):
    history = model.fit({'mfcc_input': X_train_argued}, {'mfcc_output': y_train_argued}, epochs=epoch, validation_split=0.2, batch_size=110)
    model.save_weights(weights)
    return history

def train(model, weights, epoch):
    # 生成训练数据
    X_train_argued, y_train_argued = argument_data(X_train,y_train, 300, 400, 1290, 40)
    X_train_argued, y_train_argued = get_argument_data_2d(X_train_argued, y_train_argued)
    history = train_only_mfcc(model, weights, epoch, X_train_argued, y_train_argued)
    plot_curve(history)


model = get_model()
model.compile(optimizer=optimizers.adam(lr=1e-4, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# train(model, '../weights/pre_process_mfcc.hdf5', epoch=50)

model.load_weights('../weights/pre_process_mfcc.hdf5')
print(model.evaluate(np.expand_dims(X_test,axis=1), y_test))