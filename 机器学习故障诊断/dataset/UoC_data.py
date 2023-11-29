'''
Cao P, Zhang S, Tang J. Preprocessing-free gear fault diagnosis using small datasets with deep convolutional neural network-based transfer learning[J]. Ieee Access, 2018, 6: 26241-26253.
'''

import numpy as np
from scipy.io import loadmat
import os
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(root, normalization, noise, snr, train_size, window_size, overlap, sample_num):

    mat_name = ['DataForClassification_Stage0.mat','DataForClassification_TimeDomain.mat']
    file = loadmat(os.path.join(root, mat_name[1]))

    '''
    Number of gear fault types=9={'healthy','missing','crack','spall','chip5a','chip4a','chip3a','chip2a','chip1a'}
    Number of samples per type=104
    Number of total samples=9x104=903
    The data are collected in sequence, the first 104 samples are healthy, 105th ~208th samples are missing, and etc.
    '''
    all_data = file['AccTimeDomain']  # shape: (3600,936) --- 104*9 = 936

    each_class_num = 104
    start = 0
    end = each_class_num

    all_data = all_data.T  # shape: (936,3600)

    data = [[],[],[],[],[],[],[],[],[]]  #创建一个空列表存放九种故障信号
    data_index = 0
    while end <= all_data.shape[0]:
        data[data_index].append(all_data[start:end])
        start += each_class_num
        end += each_class_num
        data_index += 1

    data = np.array(data).squeeze(axis=1)  #shape: (9,104,3600)
    data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])  #shape: (9,374400)

    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0], data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1] // window_size, window_size))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size,overlap=overlap)

    sample_data = sample_data[:, :sample_num, :]
    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    dataset = []
    labels = []
    # 计算时域特征和频域特征
    for label, signal in enumerate(norm_data):
        # calculating feature
        time_fea = time_feature(signal)  # 获得时域特征
        fft_signal = FFT(signal)  # 通过FFT获得频域信号
        frequency_fea = frequency_feature(fft_signal)

        fea = np.concatenate((time_fea, frequency_fea), axis=-1)
        dataset.append(fea)
        labels.append(np.repeat(label, fea.shape[0]))

    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0] * labels.shape[1]))
    dataset = np.array(dataset)
    dataset = dataset.reshape((dataset.shape[0] * dataset.shape[1], dataset.shape[2]))

    train_x, test_x, train_y, test_y = train_test_split(dataset, labels, train_size=train_size, shuffle=True,stratify=labels)

    return train_x, test_x, train_y, test_y