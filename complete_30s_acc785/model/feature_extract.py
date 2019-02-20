# -*- coding: utf-8 -*-
# @Time    : 2019/1/30 10:45
# @Author  : shaoeric
# @Email   : shaoeric@foxmail.com

import librosa
import os
from sklearn.preprocessing import minmax_scale
import numpy as np
from python_speech_features.base import logfbank, delta

root_path = 'genres'
target = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}


def get_fbank_feature(wavsignal, fs):
    """
    输入为wav文件数学表示和采样频率，输出为语音的FBANK特征+一阶差分+二阶差分
    :param wavsignal:
    :param fs:
    :return:
    """
    feat_fbank = logfbank(wavsignal, fs, nfilt=40, nfft=2048, winstep=0.025, winlen=0.05)
    feat_fbank_d = delta(feat_fbank, 2)
    feat_fbank_dd = delta(feat_fbank_d, 2)
    wav_feature = np.column_stack((feat_fbank, feat_fbank_d, feat_fbank_dd))
    return wav_feature


def pre_process(y, alpha=0.08, beta=0.05):
    """
    略微加强低频信号，并且使零频区域变为低频，不改变高频
    :param y: 音频信号
    :param alpha: 阈值比率，相对于极差的百分比
    :param beta: 预处理的力度
    :return: 处理后的信号
    """
    S = np.max(y) - np.min(y)
    thred = alpha * S
    for i in range(y.shape[0] - 1):
        if np.abs(y[i + 1] - y[i]) < thred:
            if y[i + 1] > y[i]:
                y[i + 1] = (1 - beta) * y[i + 1]
                y[i] = (1 + beta) * y[i]
            else:
                y[i + 1] = (1 + beta) * y[i + 1]
                y[i] = (1 - beta) * y[i]
            i += 1  # 跳过被修改过的位置
    return y


def get_feature(feature):
    Feature = []
    for root, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            y, sr = librosa.load(filename)
            y = y[: 660000]

            if feature == 'pre_process_mfcc':
                mfcc = librosa.feature.mfcc(y=pre_process(y), n_mfcc=40)
                Feature.append(mfcc.T)

            elif feature == 'concat_logbank':
                f = get_fbank_feature(y, sr)
                Feature.append(f + np.abs(f.min()))  # (1197, 40)

            elif feature == 'target':
                Feature.append(target[name.split('.')[0]])

            else:
                raise NameError
    np.save('{}.npy'.format(feature), np.array(Feature))


# get_feature('pre_process_mfcc')

