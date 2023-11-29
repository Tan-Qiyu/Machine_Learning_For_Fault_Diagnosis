'''
[1] S. Liu, J. Chen, S. He, Z. Shi, and Z. Zhou, “Subspace Network with Shared Representation learning for intelligent fault diagnosis of machine under speed transient conditions with few samples,” ISA Transactions, Oct. 2021, doi: 10.1016/j.isatra.2021.10.025.
[2] Z. Shi, J. Chen, Y. Zi, and Z. Zhou, “A Novel Multitask Adversarial Network via Redundant Lifting for Multicomponent Intelligent Fault Detection Under Sharp Speed Variation,” IEEE Transactions on Instrumentation and Measurement, vol. 70, pp. 1–10, 2021, doi: 10.1109/tim.2021.3055821.
'''

import numpy as np
import os
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

def feature_data_prepare(root, normalization, noise, snr, train_size, window_size, overlap, sample_num):
        dir = ['NC','IF_1','IF_2','IF_3','OF_1','OF_2','OF_3']  #不同损伤程度的故障类型名
        txt_name = [['REC3642_ch2.txt','REC3643_ch2.txt','REC3644_ch2.txt','REC3645_ch2.txt','REC3646_ch2.txt','REC3647_ch2.txt','REC3648_ch2.txt','REC3649_ch2.txt','REC3650_ch2.txt'],
                    ['REC3597_ch2.txt','REC3598_ch2.txt','REC3599_ch2.txt','REC3600_ch2.txt','REC3601_ch2.txt','REC3602_ch2.txt','REC3603_ch2.txt','REC3604_ch2.txt','REC3605_ch2.txt','REC3606_ch2.txt'],
                    ['REC3619_ch2.txt','REC3620_ch2.txt','REC3621_ch2.txt','REC3623_ch2.txt','REC3624_ch2.txt','REC3625_ch2.txt','REC3626_ch2.txt','REC3627_ch2.txt','REC3628_ch2.txt'],
                    ['REC3532_ch2.txt','REC3533_ch2.txt','REC3534_ch2.txt','REC3535_ch2.txt','REC3536_ch2.txt','REC3537_ch2.txt'],
                    ['REC3513_ch2.txt','REC3514_ch2.txt','REC3515_ch2.txt','REC3516_ch2.txt','REC3517_ch2.txt','REC3518_ch2.txt'],
                    ['REC3494_ch2.txt','REC3495_ch2.txt','REC3496_ch2.txt','REC3497_ch2.txt','REC3498_ch2.txt','REC3499_ch2.txt'],
                    ['REC3476_ch2.txt','REC3477_ch2.txt','REC3478_ch2.txt','REC3479_ch2.txt','REC3480_ch2.txt','REC3481_ch2.txt']]

        txt_index = [0,0,0,0,0,0,0]  #元素对应每一个txt文件位置
        data1 = [[],[],[],[],[],[],[]]
        for num, each_dir in enumerate(dir):
            with open(os.path.join(root,each_dir,txt_name[num][txt_index[num]])) as file:
                for line in file.readlines()[16:]:  #前16行说明不读取
                    line = line.strip('\n')  #删除末尾的换行
                    line = line.split('\t')
                    line = list(map(lambda x:float(x),line))
                    data1[num].append(line)

        min_value = min(list(map(lambda x:len(x),data1)))

        data = [[],[],[],[],[],[],[]]
        for data1_index in range(len(data1)):
            data[data1_index] = data1[data1_index][:min_value]

        data = np.array(data)  #shape : (7,min_value,2)  --- egg:min_value = 460800 ; 第三个维度2 表示 时间+振动信号幅值
        data = data[:,:,1]  #振动信号 --- shape : (7,min_value)

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