from scipy import stats
import numpy as np

#calculating time feature
def time_feature(vibration_siganl):
    '''
    :param vibration_siganl: the vibration singal after sliding windows sampling  ---- shape:(samples num,windows size)
    :return time_fea: the related time features of vibration signal ----shape:(samples num,the num of time features)
    '''
    f1 = vibration_siganl.max(axis=1)  #时域幅值最大值
    f2 = vibration_siganl.min(axis=1)  #时域幅值最小值
    f3 = f1 - f2  #时域幅值峰峰值
    f4 = vibration_siganl.mean(axis=1)  #时域幅值平均值
    f5 = np.sqrt((vibration_siganl ** 2).sum(axis=1) / vibration_siganl.shape[1])  #时域幅值均方根
    f6 = np.abs(vibration_siganl).sum(axis=1) / vibration_siganl.shape[1]  #幅值整流平均值
    f7 = (np.sqrt(np.abs(vibration_siganl)).sum(axis=1) / vibration_siganl.shape[1]) ** 2  #时域幅值方根
    f8 = vibration_siganl.std(axis=1) ** 2  #时域幅值方差
    f9 = vibration_siganl.std(axis=1)  #时域幅值标准差
    f10 = []  #峭度
    f11 = []  # 偏度
    for each_signal in vibration_siganl:
        f10.append(stats.kurtosis(each_signal,fisher=False))  #计算峭度
        f11.append(stats.skew(each_signal))  #计算偏度
    f10 = np.array(f10)
    f11 = np.array(f11)
    f0 = np.abs(vibration_siganl).max(axis=1) # 最大绝对值
    f12 = f0 / f5  #峰值因子
    f13 = f0 / f6  #脉冲因子
    f14 = f0 / f7  #裕度因子
    f15 = f5 / f6  #波形因子

    time_fea = np.concatenate((f1.reshape(f1.shape[0],1),
                               f2.reshape(f2.shape[0],1),
                               f3.reshape(f3.shape[0],1),
                               f4.reshape(f4.shape[0],1),
                               f5.reshape(f5.shape[0],1),
                               f6.reshape(f6.shape[0],1),
                               f7.reshape(f7.shape[0],1),
                               f8.reshape(f8.shape[0],1),
                               f9.reshape(f9.shape[0],1),
                               f10.reshape(f10.shape[0],1),
                               f11.reshape(f11.shape[0],1),
                               f12.reshape(f12.shape[0],1),
                               f13.reshape(f13.shape[0],1),
                               f14.reshape(f14.shape[0],1),
                               f15.reshape(f15.shape[0],1)),axis=-1)

    return time_fea

#calculating frequency feature
def frequency_feature(fft_siganl):
    '''
    :param fft_siganl: the frequecy singal of sliding windows sampling  ---- shape:(samples num,windows size)
    :return frequency_fea: the related time features of frequency signal ----shape:(samples num,the num of frequency features)
    '''
    f1 = fft_siganl.max(axis=1)  #频域幅值最大值
    f2 = fft_siganl.min(axis=1)  #频域幅值最小值
    f3 = np.median(fft_siganl,axis=1)   #频域幅值中位数
    f4 = fft_siganl.mean(axis=1)   #频域幅值平均值
    f5 = f1 - f2  #频域幅值峰峰值
    f6 = [] #频率中心
    f7 = [] #均方频率
    f8 = []  # 均方根频率
    f9 = []  #频率方差

    frequency = np.arange(1,fft_siganl.shape[1]+1)
    for num,each_signal in enumerate(fft_siganl):
        f6.append((frequency * each_signal).sum() / each_signal.sum())  #求频率中心
        f7.append(np.sqrt(((frequency**2) * each_signal).sum() / each_signal.sum()))  # 求均方频率
        f8.append(((frequency ** 2) * each_signal).sum() / each_signal.sum())  # 求均方根频率
        f9.append((((frequency - f6[num]) ** 2) * each_signal).sum() / each_signal.sum())  #求频率方差

    f6 = np.array(f6)
    f7 = np.array(f7)
    f8 = np.array(f8)
    f9 = np.array(f9)
    f10 = np.sqrt(f9)  #频率标准差

    frequency_fea = np.concatenate((f1.reshape(f1.shape[0], 1),
                               f2.reshape(f2.shape[0], 1),
                               f3.reshape(f3.shape[0], 1),
                               f4.reshape(f4.shape[0], 1),
                               f5.reshape(f5.shape[0], 1),
                               f6.reshape(f6.shape[0], 1),
                               f7.reshape(f7.shape[0], 1),
                               f8.reshape(f8.shape[0], 1),
                               f9.reshape(f9.shape[0], 1),
                               f10.reshape(f10.shape[0],1)),axis=-1)

    return frequency_fea

#calculating time-frequency feature
def time_frequency_feature(fft_siganl):
    pass
