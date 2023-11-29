'''
Saufi S R, Ahmad Z A B, Leong M S, et al. Gearbox fault diagnosis using a deep learning model with limited data sample[J]. IEEE Transactions on Industrial Informatics, 2020, 16(10): 6263-6271.
'''

from scipy.io import loadmat
import numpy as np
import os
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(root,fault_mode, normalization, noise, snr, train_size, window_size, overlap, sample_num):

    dir = ['data2', 'data1']  # data2 : 6 health state signal file;  data1 : 11 health state signal file
    mat_name = [['case_1.mat', 'case_2.mat', 'case_3.mat', 'case_4.mat', 'case_5.mat', 'case_6.mat'],
                ['case_1.mat', 'case_2.mat', 'case_3.mat', 'case_4.mat', 'case_5.mat', 'case_6.mat', 'case_7.mat','case_8.mat', 'case_9.mat', 'case_10.mat', 'case_11.mat']]

    if fault_mode == 17:  #17分类
        data = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]  #创建一个长度为17的列表存放17种状态信号
        data_index = 0
        for num, each_dir in enumerate(dir):
            each_mat = mat_name[num]
            for each_class in each_mat:
                file = loadmat(os.path.join(root,each_dir,each_class))
                data[data_index].append(file['gs'].squeeze(axis=1))

                data_index = data_index + 1

        data = np.array(data).squeeze(axis=1)  #shape: (17,585936)

    elif fault_mode == 2:  #2分类
        data = [[],[]]  #创建一个长度为2的列表存放2种状态信号
        data_index = 0
        for num, each_dir in enumerate(dir):
            each_mat = mat_name[num]
            for each_class in each_mat:
                file = loadmat(os.path.join(root,each_dir,each_class))
                data[data_index].extend(file['gs'].squeeze(axis=1))

            data_index = data_index + 1

        data1 = [[],[]]
        data1[0], data1[1] = data[0], data[1][:len(data[0])]

        data = np.array(data1)  #shape: (17,585936)

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

