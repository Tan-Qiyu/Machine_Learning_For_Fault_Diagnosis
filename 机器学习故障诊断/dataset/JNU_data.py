import numpy as np
import os
from itertools import islice
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(root, normalization, noise, snr, train_size, window_size, overlap, sample_num):
    health = ['n600_3_2.csv','n800_3_2.csv','n1000_3_2.csv']  #600 800 1000转速下的正常信号
    inner = ['ib600_2.csv','ib800_2.csv','ib1000_2.csv']  #600 800 1000转速下的内圈故障信号
    outer = ['ob600_2.csv','ob800_2.csv','ob1000_2.csv']  #600 800 1000转速下的外圈故障信号
    ball = ['tb600_2.csv','tb800_2.csv','tb1000_2.csv']   #600 800 1000转速下的滚动体故障信号

    file_name = []  #存放三种转速下、四种故障状态的文件名，一共12种类型
    file_name.extend(health)
    file_name.extend(inner)
    file_name.extend(outer)
    file_name.extend(ball)

    data1 = [[],[],[],[],[],[],[],[],[],[],[],[]]  #创建一个长度为12的空列表存放12种故障数据(每一类数据不平衡)
    for num, each_name in enumerate(file_name):
        dir = os.path.join(root,each_name)
        with open(dir, "r", encoding='gb18030', errors='ignore') as f:
            for line in f:
                line = float(line.strip('\n'))  #删除每一行后的换行符号，并将字符型转化为数字
                data1[num].append(line)  #将取出来的数据逐个存放到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据（每一类数据平衡）shape：(12,500500)
    for data1_i in range(len(data1)):
        data[data1_i].append(data1[data1_i][:500500])  #将所有类型数据总长度截取为500500

    data = np.array(data).squeeze(axis=1)  #shape：(12,500500)

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