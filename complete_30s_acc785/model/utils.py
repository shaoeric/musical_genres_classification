# -*- coding: utf-8 -*-
# @Time    : 2019/2/1 12:32
# @Author  : shaoeric
# @Email   : shaoeric@foxmail.com
import matplotlib.pyplot as plt
from random import randint, sample
import numpy as np
from copy import deepcopy


def evaluate_model(model, model_weights, X_test, y_test):
    model.load_weights(model_weights)
    evaluate = model.evaluate(X_test, y_test)
    print(evaluate)

def plot_curve(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()

# 数据增强
# 随机取噪音位置以及噪音个数
def get_pos(min_num, max_num, timestep, features_num):
    """
    :return: 返回选择的点位置
    """
    num = randint(min_num, max_num)
    select_x, select_y, n = np.random.randint(0, timestep, num), np.random.randint(0, features_num, num), np.random.randint(-3, 3, num)/20+1e-5
    # select_x, select_y, n = np.random.randint(0, timestep, num), np.random.randint(0, features_num, num), np.random.randint(-2, 2, num)+1e-5
    return (select_x, select_y, n)


# 每张原图加min~max个噪音， 每张图扩充5倍
def argument_data(X_train, y_train, min_num, max_num, timestep, features_num):
    new_X, new_Y = [], []
    for i in range(X_train.shape[0]):
        new_X.append(X_train[i])
        new_Y.append(y_train[i])
        for j in range(5):
            temp = deepcopy(X_train[i])
            a, b, n = get_pos(min_num, max_num, timestep, features_num)
            try:
                temp[a, b] += n
            except:
                temp[b] += n
            new_X.append(temp)
            new_Y.append(y_train[i])
    return np.array(new_X), np.array(new_Y)

# x = np.array([[1,2,3],[2,3,4]],dtype='float')
# y = np.array([0, 1])
# print(x, y)
# # x, y = argument_data(x, y, 0, 3, 1, 3)
# print(x.shape)
# x = x.reshape(x.shape[0],-1, 1)
# print(x, y)
# print(x.shape)
# # print(x, y)
# # print(x[0].shape, x[0])


def get_argument_data_1d(X_train_argued, y_train_argued):
    anchor = sample(list(range(y_train_argued.shape[0])), y_train_argued.shape[0])
    X_train_argued, y_train_argued = X_train_argued[anchor], y_train_argued[anchor]
    return X_train_argued, y_train_argued



def get_argument_data_2d(X_train_argued, y_train_argued):
    anchor = sample(list(range(y_train_argued.shape[0])), y_train_argued.shape[0])
    X_train_argued, y_train_argued = np.expand_dims(X_train_argued[anchor], axis=1), y_train_argued[anchor]
    # X_train_argued, y_train_argued = X_train_argued[anchor], y_train_argued[anchor]
    return X_train_argued, y_train_argued
