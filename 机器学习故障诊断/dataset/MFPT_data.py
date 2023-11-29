from scipy.io import loadmat
import numpy as np
import os
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(root, normalization, noise, snr, train_size, window_size, overlap, sample_num):
    dir = ['1 - Three Baseline Conditions','3 - Seven More Outer Race Fault Conditions','4 - Seven Inner Race Fault Conditions']
    mat_name = [['baseline_1.mat'],
                ['OuterRaceFault_vload_1.mat','OuterRaceFault_vload_2.mat','OuterRaceFault_vload_3.mat','OuterRaceFault_vload_4.mat',
                 'OuterRaceFault_vload_5.mat','OuterRaceFault_vload_6.mat','OuterRaceFault_vload_7.mat'],
                ['InnerRaceFault_vload_1.mat','InnerRaceFault_vload_2.mat','InnerRaceFault_vload_3.mat',
                 'InnerRaceFault_vload_4.mat','InnerRaceFault_vload_5.mat','InnerRaceFault_vload_6.mat','InnerRaceFault_vload_7.mat']]

    data = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    data_index = 0
    for num, each_dir in enumerate(dir):
        for each_mat in mat_name[num]:
            f = loadmat(os.path.join(root,each_dir,each_mat))
            if num == 0:  #num=0时为正常信号
                data[data_index].append(f['bearing'][0][0][1].squeeze(axis=1)[:146484])  #取正常样本前146484，使得与其余故障样本数平衡
            else:
                data[data_index].append(f['bearing'][0][0][2].squeeze(axis=1))

            data_index = data_index + 1

    data = np.array(data).squeeze(axis=1)  #shape:(15,146484)

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


