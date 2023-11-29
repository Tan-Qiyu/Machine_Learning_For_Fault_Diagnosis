import numpy as np
import os
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(root, normalization, noise, snr, train_size, window_size, overlap, sample_num):
    csv_name = ['train.csv','test_data.csv']

    data1 = [[],[],[],[],[],[],[],[],[],[]]
    with open(os.path.join(root,csv_name[0]),encoding='gbk') as file:
        for line in file.readlines()[1:]:  #从第二行开始读取数据
            line = line.split(',')[1:]  #用逗号分隔数据，并舍弃第一列的编号id  6001 ---6000 data + 1label
            line = list(map(lambda x:float(x), line))  #将6001个字符转化为数字
            data1[int(line[-1])].append(line[:-1])  #按照标签将6000个数据存到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], []]
    for data1_index in range(len(data1)):
        data[data1_index].append(data1[data1_index][:43])

    data = np.array(data).squeeze(axis=1)  #shape: (10,43,6000)
    data = data[:,:,:(data.shape[2] // window_size) * window_size]  #shape: (10,43,5120)  when window_size == 1024

    data = data.reshape((data.shape[0],data.shape[1],data.shape[2] // window_size,window_size))  #将5120个数据按照window_size划分  shape: (10,43,5,1024)
    data = data.reshape((data.shape[0],data.shape[1]*data.shape[2],data.shape[3]))  #shape: (10,215,1024)
    data = data.reshape((data.shape[0],data.shape[1]*data.shape[2]))  #shape: (10,215*1024)

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