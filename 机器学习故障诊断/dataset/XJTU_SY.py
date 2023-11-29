'''
数据集详细说明及引用：
引用格式：
[1]雷亚国,韩天宇,王彪,李乃鹏,闫涛,杨军.XJTU-SY滚动轴承加速寿命试验数据集解读[J].机械工程学报,2019,55(16):1-6.
说明：
XJTU-SY bearing datasets；
包括3种工况：35Hz12kN、37.5Hz11kN、40Hz10kN；
每种工况数据下分别包括5种不同的故障，详见文献[1]中的表3；
不同的故障下包括了多次采样的振动信号数据，每个样本代表1分钟内采样的振动信号，名称表示采样时间，
可以看出时间越靠后，就越接近故障发生的时间
每个样本包括利用加速度传感器在水平轴和垂直轴两个方向采集的振动信号，每分钟包括32768个采样点
'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(dataset_path,minute_value,channel,sample_number,train_size,window_size,overlap,normalization,noise,snr):
    '''
    :param dataset_path:  数据集目录
    :param minute_value:  取最后csv文件信号的个数
    :param channel:  振动信号的通道
    :param sample_number:  每种 故障类型的样本数
    :param train_size: 训练集比例
    :param window_size: 每个样本长度
    :param overlap:  滑窗偏移量
    :param normalization:  归一化
    :param noise:  添加噪声
    :param snr: 信噪比
    :return:  训练集、测试集 特征及标签
    '''

    '''
    # 样本标签
    label1 = [i for i in range(0, 5)]  # 工况1：35Hz12kN
    label2 = [i for i in range(5, 10)]  # 工况2：37.5Hz11kN
    label3 = [i for i in range(10, 15)]  # 工况3：40Hz10kN
    '''

    WC = os.listdir(dataset_path)  # 遍历根目录，目录下包括三种工况的文件夹名

    datasetname1 = os.listdir(os.path.join(dataset_path, WC[0]))
    datasetname2 = os.listdir(os.path.join(dataset_path, WC[1]))
    datasetname3 = os.listdir(os.path.join(dataset_path, WC[2]))

    data_name = []
    data_name.extend(datasetname1)
    data_name.extend(datasetname2)
    data_name.extend(datasetname3)

    # 工况1数据及标签
    data = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in tqdm(range(len(data_name))):
        if i>= 0 and i <=4:
            dir = os.path.join('/tmp', dataset_path, WC[0], data_name[i])   # 工况1：35Hz12kN
            files = os.listdir(dir)
        elif i > 4 and i <= 9:
            dir = os.path.join('/tmp', dataset_path, WC[1], data_name[i])   # 工况2：37.5Hz11kN
            files = os.listdir(dir)
        elif i > 9 and i <= 14:
            dir = os.path.join('/tmp', dataset_path, WC[2], data_name[i])  # 工况3：40Hz10kN
            files = os.listdir(dir)

        # 提取振动信号最后故障时刻的采样值时，csv文件的读取顺序不是按照数字排序进行读取的，例如123在99之前，故根据.csv文件的前缀名进行读取
        files_list = list(map(lambda x: int(x[:-4]), files))
        load_file_name = list(range(np.array(files_list).max() - minute_value + 1,np.array(files_list).max() + 1))  # 取出最后minute_value分钟内需要处理的故障数据
        load_file_name = list(map(lambda y: str(y) + '.csv', load_file_name))

        data11 = np.empty((0,))
        for ii in range(minute_value):  # Take the data of the last three CSV files
            path1 = os.path.join(dir, load_file_name[ii])
            fl = pd.read_csv(path1)
            if channel == 'X':  # 水平轴信号
                fl = fl["Horizontal_vibration_signals"]
            elif channel == 'Y':  # 垂直轴信号
                fl = fl["Vertical_vibration_signals"]
            elif channel == 'XY':  # 水平轴和垂直轴信号
                fl = fl
            else:
                print('the vibration signal with this channel is not exsisted!')

            fl = fl.values
            data11 = np.concatenate((data11,fl),axis=0)

        data[i].append(data11)

    data = np.array(data)
    data = data.squeeze(axis=1)



    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0], data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1]//window_size, window_size))
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
