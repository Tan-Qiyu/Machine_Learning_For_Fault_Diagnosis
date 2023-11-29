import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(dataset_path, rpm, channel, noise,snr,window_size,overlap,sample_number,normalization,train_size):
    '''
    :param dataset_path: 数据集路径
    :param rpm: 电机转速
    :param channel: 通道信号 0-8
    :param noise:  添加噪声
    :param snr: 信噪比
    :param window_size: 每个样本长度
    :param overlap: 滑窗偏移量
    :param sample_number: 每种故障类型样本数
    :param normalization: 归一化
    :param train_size: 训练集比例
    :return: 训练集。测试集特征及标签
    '''

    dir = ['正常运行下','点蚀','点蚀磨损','断齿','断齿、磨损混合故障','磨损']
    if rpm == 880:
        txt_name = [['normal880.txt','normal880-1.txt','normal880-2.txt','normal880-3.txt'],
                    ['dianshi880.txt','dianshi880-1.txt','dianshi880-2.txt','dianshi880-3.txt'],
                    ['dianmo880.txt','dianmo880-1.txt','dianmo880-2.txt','dianmo880-3.txt'],
                    ['duanchi880.txt','duanchi880-1.txt','duanchi880-2.txt','duanchi880-3.txt'],
                    ['duanmo880.txt','duanmo880-1.txt','duanmo880-2.txt','duanmo880-3.txt'],
                    ['mosun880.txt','mosun880-1.txt','mosun880-2.txt','mosun880-3.txt']]
    elif rpm == 1470:
        txt_name = [['normal1500.txt'],
                    ['dianshi1470.txt','dianshi1500.txt'],
                    ['dianmo1470.txt'],
                    ['duanchi1500.txt'],
                    ['duanmo1470.txt'],
                    ['mosun1470.txt']]


    data = []
    for dir_num, each_dir in enumerate(dir):
        sub_txt = txt_name[dir_num]
        subdata = []
        for each_txt in sub_txt:
            file_name = os.path.join(dataset_path,each_dir,each_txt)
            with open(file_name,encoding='gbk') as f:
                for line in f.readlines()[1:]:  #从第二行开始读取，第一行为空格
                    line = line.strip('\t\n')  #删除每一行的最后一个制表符和换行符号
                    line = line.split('\t')  #按制表符\t进行分隔
                    subdata.append(list(map(lambda x: float(x), line)))   #将字符数据转化为数字

        subdata = np.array(subdata)  #shape:(样本数，通道数)
        data.append(subdata)

    sam_list = []
    list(map(lambda x: sam_list.append(x.shape[0]), data))
    sam_min = min(sam_list)
    max_min = max(sam_list)
    if sam_min != max_min:
        balance_data = [[], [], [], [], [], []]
        for all_data_index, class_data in enumerate(data):
            # np.random.shuffle(class_data)
            balance_data[all_data_index].append(data[all_data_index][:sam_min, :])
        data = np.array(balance_data).squeeze(axis=1)

    data = np.array(data)   #shape:(故障类型数，样本数，通道数)
    data = data[:,:,channel]  #选择通道

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

    sample_data = sample_data[:, :sample_number, :]

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