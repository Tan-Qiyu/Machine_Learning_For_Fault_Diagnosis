'''
Time: 2022/10/07 00:42
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/Machine_Learning_For_Fault_Diagnosis
Dataset Download Link：https://github.com/Tan-Qiyu/Mechanical_Fault_Diagnosis_Dataset
引用格式:[1]谭启瑜,马萍,张宏立.基于图卷积神经网络的滚动轴承故障诊断[J].噪声与振动控制,2023,43(06):101-108+116.
'''

'''
--------------------------------------------------参数介绍------------------------------------------
dataset_name： 数据集名称
    CWRU、SEU（bearing、gear）、XJTU、Bearing_311、QianPeng_gear、JNU、MFPT、Wind_Gearbox、UoC、DC、SQV 共11个公开数据集
    
dataset_path： 数据集目录地址
    CWRU  "E:\故障诊断数据集\凯斯西储大学数据" -------------------------------------------# 凯斯西储大学轴承数据集
    SEU   "E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\bearingset" --------- -# 东南大学轴承子数据
    SEU   "E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\gearset" --------------# 东南大学齿轮子数据
    XJTU  "E:\故障诊断数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets"----# 西安交通大学-昇阳轴承退化数据集
    Bearing_311    "E:\故障诊断数据集\LW轴承实验数据2016(西安交通大学)"-----------------------# 西交试验台数据集：马萍、张西宁等
    QianPeng_gear  "E:\故障诊断数据集\齿轮箱故障数据_千鹏公司"--------------------------------# 千鹏公司齿轮箱数据集
    JNU   "E:\故障诊断数据集\江南大学轴承数据(一)\数据"-------------------------------------# 江南大学轴承数据集
    MFPT  "E:\故障诊断数据集\MFPT Fault Data Sets"--------------------------------------# 美国-机械故障预防技术学会MFPT数据集
    Wind_Gearbox   "E:\故障诊断数据集\fault-dataset_2d-collection-main\fault-dataset_2d-collection-main\HS_gear"--------# 风机齿轮箱数据集
    UoC   "E:\故障诊断数据集\美国-康涅狄格大学-齿轮数据集"------------------------------------# 美国-康涅狄格大学-齿轮数据集
    DC    "E:\故障诊断数据集\DC轴承数据"--------------------------------------------------# 中国-轴承数据集（DC竞赛）
    SQV   "E:\故障诊断数据集\SQV-public"-------------------------------------------------# SQV轴承变转速数据集

dir_path:   CWRU数据集的采样频率和传感器位置
    12DE  # 12kHZ Drive End Dataset
    48DE  # 48kHZ Drive End Dataset

SEU_channel:  SEU数据集的数据通道
    0、1、2、3、4、5、6、7  共8个通道

minute_value：  XJTU-SY数据集使用最后多少个文件数据进行实验验证

XJTU_channel：   XJTU-SY数据集的数据通道
    X Y XY  共3种通道

QianPeng_rpm:   QianPeng_gear数据集的电机转速（rpm）
    880 1470

QianPeng_channel:   QianPeng_gear数据集的信号通道
    0、1、2、3、4、5、6、7

wind_mode：  Wind_Gearbox数据集分类任务
    2 17
    
sample_num：   每一种故障类型的样本数（CWRU数据集除外，为每一种故障下的一种工况的样本数，即CWRU每一类故障样本数为sample_num * 工况数 = sample_num * 4）

train_size： 训练集比例

sample_length：  样本采样长度 = 网络的输入特征长度

overlap：   滑窗采样偏移量，当sample_length = overlap时为无重叠顺序采样

norm_type：  原始振动信号的归一化方式
    unnormalization        # 不进行归一化
    Z-score Normalization  # 均值方差归一化
    Max-Min Normalization  # 最大-最小归一化：归一化到0到1之间
    -1 1 Normalization     # 归一化到-1到1之间

noise:   是否往原始振动信号添加噪声
    0   # 不添加噪声
    1   # 添加噪声

snr：   当noise = 1时添加的噪声的信噪比大小；当noise = 0时此参数无效

model_type:  网络模型
    KNN、SVM、RF 

visualization:  是否绘制混淆矩阵与t-SNE可视化图
    True   # 进行可视化
    False  # 不进行可视化
    
save_visualization_results：  是否保存可视化结果
    True  # 保存（仅visualization = True时成立） 
    False # 不保存
    
visualization_dirpath:   可视化结果的保存路径（仅save_visualization_results = True时成立）
    保存路径为： visualization_dirpath + \混淆矩阵-dataset_name-noise-snr(年月日 时分秒).tif'  # 混淆矩阵
               visualization_dirpath + \时频域特征可视化-dataset_name-noise-snr(年月日 时分秒).tif'  # tsne特征可视化
    
save_data：   是否保存多次实验结果并存为excel文件
    True   # 保存
    False  # 不保存
    
save_data_dirpath：   多次实验结果存为excel文件的路径
    保存路径：  args.save_data_dirpath + "\dataset_name-noise-snr(年月日 时分秒).xlsx".format(save_name)

'''
import argparse
from model import ML_classifier
from 机器学习故障诊断.model.save_data import save_data

def parse_args():
    parser = argparse.ArgumentParser()
    # basic parameters
    # ===================================================dataset_2d parameters============================================================================
    parser.add_argument('--dataset_name', type=str, default='SEU', help='the name of the dataset_2d')
    parser.add_argument('--dataset_path', type=str, default=r"E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\bearingset", help='the file path of the dataset_2d')
    parser.add_argument('--dir_path', type=str, default='12DE', help='CWRU：12DE ; 48DE')
    parser.add_argument('--SEU_channel', type=int, default=1, help='SEU channel signal：0-7')
    parser.add_argument('--minute_value', type=int, default=0, help='the last (minute_value) csv file of XJTU datasets each fault class')
    parser.add_argument('--XJTU_channel', type=str, default='X', help='XJTU channel signal:X 、Y 、XY')
    parser.add_argument('--QianPeng_rpm', type=int, default=880, help='the motor rpm of QianPeng_gear dataset_2d--880 or 1470')
    parser.add_argument('--QianPeng_channel', type=int, default=3,help='the signal channel of QianPeng_gear dataset_2d--0-8')
    parser.add_argument('--wind_mode', type=int, default=17,help='Wind_Gearbox mode-- 17 fault class / 2 fault class')

    # ===================================================data preprocessing parameters========================================================================
    parser.add_argument('--sample_num', type=int, default=200, help='the number of samples')
    parser.add_argument('--train_size', type=float, default=0.6, help='train size')
    parser.add_argument('--sample_length', type=int, default=1024, help='the length of each samples')
    parser.add_argument('--overlap', type=int, default=1024, help='the sampling shift of neibor two samples')
    parser.add_argument('--norm_type', type=str, default='unnormalization', help='unnormalization、Z-score Normalization、Max-Min Normalization、-1 1 Normalization')
    parser.add_argument('--noise', type=int, default=1, help='whether add noise')
    parser.add_argument('--snr', type=int, default=-8, help='the snr of noise')

    # ===================================================model parameters=============================================================================
    parser.add_argument('--model_type', type=str, default='RF', help='分类器：KNN ; SVM ; RF')

    # ===================================================visualization parameters=============================================================================
    parser.add_argument('--visualization', type=bool, default=False, help='whether visualize')
    parser.add_argument('--save_visualization_results', type=bool, default=False, help='whether save visualization results')
    parser.add_argument('--visualization_dirpath', type=str, default=r"C:\Users\Administrator\Desktop\故障诊断开源代码\results\save_visualization",help='the save dirpath of visualization')
    parser.add_argument('--save_data', type=bool, default=False,help='whether save data')
    parser.add_argument('--save_data_dirpath', type=str, default=r"C:\Users\Administrator\Desktop\故障诊断开源代码\results\save_data", help='the dirpath of saved data')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    if args.save_data == False:
        acc = ML_classifier.train_utils(args)
    else:
        save_data(args,trials=10)


