import numpy as np
import os
from itertools import islice
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature


def feature_data_prepare(root,channel,noise,snr,normalization,window_size,overlap,sample_number,train_size):

    if "bearingset" in root:  # Path of bearingset
        # Data names of 5 bearing fault types under two working conditions
        data_name = ["ball_20_0.csv", "comb_20_0.csv", "health_20_0.csv", "inner_20_0.csv", "outer_20_0.csv",
                     "ball_30_2.csv", "comb_30_2.csv", "health_30_2.csv", "inner_30_2.csv", "outer_30_2.csv"]

    elif "\gearset" in root:  # Path of gearset
        # Data names of 5 gear fault types under two working conditions
        data_name = ["Chipped_20_0.csv", "Health_20_0.csv", "Miss_20_0.csv", "Root_20_0.csv", "Surface_20_0.csv",
                     "Chipped_30_2.csv", "Health_30_2.csv", "Miss_30_2.csv", "Root_30_2.csv", "Surface_30_2.csv"]

    data_list = [[], [], [], [], [], [], [], [], [], []]  # 创建10个空列表存放十种故障数据

    for num_state, dir_name in enumerate(data_name):

        dir = os.path.join(root, dir_name)
        b_g_data = open(dir, "r", encoding='gb18030', errors='ignore')

        i = 0

        if dir_name == 'ball_20_0.csv':
            for line in islice(b_g_data, 16, None):  # 逐行取出，移除前16行说明
                if i < sample_number * window_size:  # 限制信号长度，因为信号总长为1048560，8传感器，10类型，即1048560*8*10接近上亿的数值会对计算机造成一定的影响
                    line = line.rstrip()  # 移除每一行末尾的字符
                    word = line.split(",", 8)  # 使用8个,进行分隔
                    word = list(map(float, word[:-1]))  # 将每行的str数据转化为float数值型，并移除最后一个空元素
                    data_list[num_state].append(word)  # 将故障信号存在相应的列表内
                    i += 1
                else:
                    break

        else:
            for line in islice(b_g_data, 16, None):
                if i < sample_number * window_size:
                    line = line.rstrip()
                    word = line.split("\t", 8)
                    word = list(map(float, word))
                    data_list[num_state].append(word)
                    i += 1
                else:

                    break

    # 振动信号
    all_data = np.array(data_list)  # the dimention of bearing is: classes * sample_length * channels

    data = all_data[:, :, channel]

    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0],data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data


    #滑窗采样
    sample_data = np.zeros((noise_data.shape[0],noise_data.shape[1] // window_size,window_size))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size, overlap=overlap)

    sample_data = sample_data[:,:sample_number,:]
    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0],sample_data.shape[1],sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    dataset = []
    labels = []
    #计算时域特征和频域特征
    for label,signal in enumerate(norm_data):
        # calculating feature
        time_fea = time_feature(signal)  #获得时域特征
        fft_signal = FFT(signal)  #通过FFT获得频域信号
        frequency_fea = frequency_feature(fft_signal)

        fea = np.concatenate((time_fea,frequency_fea),axis=-1)
        dataset.append(fea)
        labels.append(np.repeat(label,fea.shape[0]))

    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0]*labels.shape[1]))
    dataset = np.array(dataset)
    dataset = dataset.reshape((dataset.shape[0]*dataset.shape[1],dataset.shape[2]))

    train_x, test_x, train_y,test_y = train_test_split(dataset,labels, train_size=train_size, shuffle=True,stratify=labels)

    return train_x,test_x,train_y,test_y