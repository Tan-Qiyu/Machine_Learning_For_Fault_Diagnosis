import os
import numpy as np
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(dataset_path,sample_number,train_size,window_size,overlap,normalization,noise,snr):

    file_dir = dataset_path

    file_name = ['43 正常','保持架','滚动体','内圈','外圈']

    holder_servity_name = ['40 较轻','41 较严重']
    ball_servity_name = ['34 较轻','37 较严重']
    inner_servity_name = ['29 较轻','30 较严重']
    outer_servity_name = ['25 较轻','27 较严重']

    data_name = ['c1.txt','c2.txt','c3.txt','c4.txt','c5.txt','c6.txt','c7.txt','c8.txt','c9.txt','c10.txt','c11.txt','c12.txt',]

    data = [[],[],[],[],[],[],[],[],[]]  #创建9个空列表存放9中故障类型，包括正常和较轻、较严重下的保持架故障、滚动体故障、内圈故障、外圈故障

    for each_file_name in file_name:
        if each_file_name == '43 正常':
            for each_data_name in data_name:
                file = open(os.path.join(file_dir,each_file_name,each_data_name), encoding='gbk')
                for line in file:
                    data[0].append(float(line.strip())) #正常信号存在data[0]中

        elif each_file_name == '保持架':
            for num,each_holder_servity_name in enumerate(holder_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir,each_file_name,each_holder_servity_name,each_data_name), encoding='gbk')
                    for line in file:
                        data[num+1].append(float(line.strip())) #保持架较轻故障信号存在data[1]中，保持架较严重故障信号存在data[2]中

        elif each_file_name == '滚动体':
            for num,each_ball_servity_name in enumerate(ball_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir,each_file_name,each_ball_servity_name,each_data_name), encoding='gbk')
                    for line in file:
                        data[num+3].append(float(line.strip())) #滚动体较轻故障信号存在data[3]中，保持架较严重故障信号存在data[4]中

        elif each_file_name == '内圈':
            for num,each_inner_servity_name in enumerate(inner_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir,each_file_name,each_inner_servity_name,each_data_name), encoding='gbk')
                    for line in file:
                        data[num+5].append(float(line.strip())) #内圈较轻故障信号存在data[5]中，保持架较严重故障信号存在data[6]中

        elif each_file_name == '外圈':
            for num,each_outer_servity_name in enumerate(outer_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir,each_file_name,each_outer_servity_name,each_data_name), encoding='gbk')
                    for line in file:
                        data[num+7].append(float(line.strip())) #外圈较轻故障信号存在data[7]中，保持架较严重故障信号存在data[8]中

    data = np.array(data)  #data.shpe = (9 , 20480 * 12).  9 represent 9 classes faulty type, 20480 is sample frequecy each seconds, 12 is the all time length of sample

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

    sample_data = sample_data[:,:sample_number,:]

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