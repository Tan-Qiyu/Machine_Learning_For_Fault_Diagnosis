from scipy.io import loadmat
import numpy as np
import os
from sklearn.model_selection import train_test_split
from 机器学习故障诊断.dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from 机器学习故障诊断.dataset._featuret import time_feature,frequency_feature,time_frequency_feature

'''Normal Baseline Data由于仅有4个文件97.mat、98.mat、99.mat、100.mat，
为了方便访问及处理，将四个文件都拷贝到其余三个文件夹中'''

#正常数据，负载0、1、2、3
NB = ['97.mat', '98.mat', '99.mat', '100.mat']

'''12k Drive End Bearing Fault Data'''
#内圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
IR07_12DE = ['105.mat', '106.mat', '107.mat', '108.mat']
IR14_12DE = ['169.mat', '170.mat', '171.mat', '172.mat']
IR21_12DE = ['209.mat', '210.mat', '211.mat', '212.mat']

#外圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3 in Centerd @6:00
OR07_12DE = ['130.mat', '131.mat', '132.mat', '133.mat']
OR14_12DE = ['197.mat', '198.mat', '199.mat', '200.mat']
OR21_12DE = ['234.mat', '235.mat', '236.mat', '237.mat']

#滚动体故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
B07_12DE = ['118.mat', '119.mat', '120.mat', '121.mat']
B14_12DE = ['185.mat', '186.mat', '187.mat', '188.mat']
B21_12DE = ['222.mat', '223.mat', '224.mat', '225.mat']

#全部数据，包括正常加九种不同类型、不同损伤程度的故障，一共十种
full_data_12DE = [NB,  IR07_12DE, IR14_12DE, IR21_12DE, OR07_12DE, OR14_12DE, OR21_12DE,B07_12DE, B14_12DE, B21_12DE]

'''48k Drive End Bearing Fault Data'''
#内圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
IR07_48DE = ['109.mat', '110.mat', '111.mat', '112.mat']
IR14_48DE = ['174.mat', '175.mat', '176.mat', '177.mat']
IR21_48DE = ['213.mat', '214.mat', '215.mat', '217.mat']

#外圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3 in Centerd @6:00
OR07_48DE = ['135.mat', '136.mat', '137.mat', '138.mat']
OR14_48DE = ['201.mat', '202.mat', '203.mat', '204.mat']
OR21_48DE = ['238.mat', '239.mat', '240.mat', '241.mat']

#滚动体故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
B07_48DE = ['122.mat', '123.mat', '124.mat', '125.mat']
B14_48DE = ['189.mat', '190.mat', '191.mat', '192.mat']
B21_48DE = ['226.mat', '227.mat', '228.mat', '229.mat']

#全部数据，包括正常加九种不同类型、不同损伤程度的故障，一共十种
full_data_48DE = [NB, IR07_48DE, IR14_48DE, IR21_48DE, OR07_48DE, OR14_48DE, OR21_48DE,B07_48DE, B14_48DE, B21_48DE]
full_data = [full_data_12DE,full_data_48DE]

# ================================the processing of train data and test data ========================================
def feature_data_prepare(dataset_path,sample_number,train_size,dir_path,Window_size,overlap,normalization,noise,snr):
    '''
    :param dataset_path: the file path of cwru datasets
    :param sample_number: the samples numbers of each fault type and 4 motor rpm ----total samples = samples_number * 40 ---- 40 = 10(fault classes) * 4(4 motor rpm)
    :param train_size: train sample / totlal samples
    :param dir_path: the type of vibration sensors signal(different sample frequency)  ---- 12DE: 12k Drive End Bearing Fault Data; 48DE: 48k Drive End Bearing Fault Data
    :param Window_size: the sample length of each sample
    :param overlap: the data shift of neibor two samples
    :param normalization: the type of normalization
    :param noise: add noise or don't add noise
    :param snr: the snr of noise
    :return train_x,test_x,train_y,test_y
            the features of train data(train_x,train_y),test data(test_x,test_y)
    '''

    File_dir = dataset_path

    dirname = ['Normal Baseline Data','12k Drive End Bearing Fault Data','48k Drive End Bearing Fault Data']

    if dir_path == '12DE':
        data_path = os.path.join(File_dir,dirname[1])
        file_number = 0
    elif dir_path == '48DE':
        data_path = os.path.join(File_dir, dirname[2])
        file_number = 1

    data_list = [[],[],[],[],[],[],[],[],[],[]]

    i = 0

    for bearing_state in enumerate(full_data[file_number]):

        for num,load in enumerate(bearing_state[1]):
            data = loadmat(os.path.join(data_path, load))
            if eval(load.split('.')[0]) < 100:
                vibration = data['X0' + load.split('.')[0] + '_DE_time']
            elif eval(load.split('.')[0]) == 174:
                vibration = data['X' + '173' + '_DE_time']
            else:
                vibration = data['X' + load.split('.')[0] + '_DE_time']

            #添加不同信噪比的噪声
            if noise == 'y' or noise == 1:
                vibration = Add_noise(vibration,snr).reshape(Add_noise(vibration,snr).shape[0],1)
            elif noise == 'n' or noise == 0:
                vibration = vibration

            slide_data = Slide_window_sampling(vibration, Window_size, overlap)  # 滑窗采样
            if normalization != 'unnormalization':
                data_x = Normal_signal(slide_data,normalization)  #归一化
            else:
                data_x = slide_data

            # np.random.shuffle(data_x)  #将数据shuffle

            if sample_number == 112 and num == 0:  #当每种故障下不同负载的样本数为112，总样本数为112*40=4480时，在每种故障下的0负载增加2个样本数使得样本总数为4480+2*10=4500
                data_sample_x = data_x[:sample_number + 2,:]  #时域信号
                data_list[i].append(data_sample_x)
            else:
                data_sample_x = data_x[:sample_number, :]  # 时域信号
                data_list[i].append(data_sample_x)

        i = i + 1

    all_data = []

    for sample in data_list:
        each_data = np.concatenate((sample[0],sample[1],sample[2],sample[3]),axis=0)
        all_data.append(each_data)

    dataset = []
    labels = []
    for label,signal in enumerate(all_data):
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