import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from 机器学习故障诊断.model.visualization import visualization
from 机器学习故障诊断.dataset import CWRU_data,SEU_data,XJTU_SY,Bearing_311,QianPeng_Gearbox,JNU_data,MFPT_data,Wind_Gearbox,UoC_data,DC_data,SQV_data

plt.rcParams['font.sans-serif'] = ['SimSun']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def train_utils(args):

    if args.dataset_name == 'SEU':
        if 'bearingset' in args.dataset_path:
            save_name = args.dataset_name + '-bearing-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
        elif 'gearset' in args.dataset_path:
            save_name = args.dataset_name + '-gear-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
    else:
        save_name = args.dataset_name + '-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'

    #==============================================================1、训练集、测试集===================================================

    if args.dataset_name == 'CWRU':  #凯斯西储大学轴承数据集
        train_x,test_x,train_y,test_y = CWRU_data.feature_data_prepare(dataset_path=args.dataset_path,sample_number=args.sample_num,
                                                                    train_size=args.train_size,dir_path=args.dir_path,Window_size=args.sample_length,
                                                                    overlap=args.overlap,normalization=args.norm_type,noise=args.noise,snr=args.snr)

    elif args.dataset_name == 'SEU':  #东南大学数据集
        train_x,test_x,train_y,test_y = SEU_data.feature_data_prepare(root=args.dataset_path,channel=args.SEU_channel,train_size=args.train_size,
                                                                    noise=args.noise,snr=args.snr,normalization=args.norm_type,window_size=args.sample_length,
                                                                    overlap=args.overlap,sample_number=args.sample_num)

    elif args.dataset_name == 'XJTU':  #西安交通大学轴承退化数据集
        train_x,test_x,train_y,test_y = XJTU_SY.feature_data_prepare(dataset_path=args.dataset_path, minute_value=args.minute_value,
                                                                 channel=args.XJTU_channel, sample_number=args.sample_num, train_size=args.train_size,
                                                                 window_size=args.sample_length,overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr)

    elif args.dataset_name == 'Bearing_311':  #西安交通大学轴承试验台数据——张西宁，马萍
        train_x,test_x,train_y,test_y = Bearing_311.feature_data_prepare(dataset_path=args.dataset_path, sample_number=args.sample_num, train_size=args.train_size,
                                                                window_size=args.sample_length, overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr)

    elif args.dataset_name == 'QianPeng_gear':  #千鹏公司齿轮箱数据集
        train_x,test_x,train_y,test_y = QianPeng_Gearbox.feature_data_prepare(dataset_path=args.dataset_path, rpm=args.QianPeng_rpm, channel=args.QianPeng_channel,
                                                                      noise=args.noise, snr=args.snr,window_size=args.sample_length, overlap=args.overlap,
                                                                      sample_number=args.sample_num, normalization=args.norm_type,train_size=args.train_size)

    elif args.dataset_name == 'JNU':  #江南大学轴承数据集
        train_x,test_x,train_y,test_y = JNU_data.feature_data_prepare(root=args.dataset_path, normalization=args.norm_type, noise=args.noise, snr=args.snr,
                                                                    train_size=args.train_size, window_size=args.sample_length, overlap=args.overlap, sample_num=args.sample_num)

    elif args.dataset_name == 'MFPT':  #美国-机械故障预防技术学会 MFPT 数据集
        train_x, test_x, train_y, test_y = MFPT_data.feature_data_prepare(root=args.dataset_path,normalization=args.norm_type, noise=args.noise,snr=args.snr,
                                                                        train_size=args.train_size,window_size=args.sample_length,overlap=args.overlap,sample_num=args.sample_num)

    elif args.dataset_name == 'Wind_Gearbox':
        train_x, test_x, train_y, test_y = Wind_Gearbox.feature_data_prepare(root=args.dataset_path,fault_mode=args.wind_mode,normalization=args.norm_type,noise=args.noise, snr=args.snr,
                                                                        train_size=args.train_size,window_size=args.sample_length,overlap=args.overlap,sample_num=args.sample_num)

    elif args.dataset_name == 'UoC':
        train_x, test_x, train_y, test_y = UoC_data.feature_data_prepare(root=args.dataset_path, normalization=args.norm_type, noise=args.noise, snr=args.snr,
                                                                         train_size=args.train_size, window_size=args.sample_length, overlap=args.overlap, sample_num=args.sample_num)

    elif args.dataset_name == 'DC':
        train_x, test_x, train_y, test_y = DC_data.feature_data_prepare(root=args.dataset_path, normalization=args.norm_type, noise=args.noise, snr=args.snr,
                                                                         train_size=args.train_size, window_size=args.sample_length, overlap=args.overlap, sample_num=args.sample_num)

    elif args.dataset_name == 'SQV':
        train_x, test_x, train_y, test_y = SQV_data.feature_data_prepare(root=args.dataset_path, normalization=args.norm_type, noise=args.noise, snr=args.snr,
                                                                         train_size=args.train_size, window_size=args.sample_length, overlap=args.overlap, sample_num=args.sample_num)

    else:
        print('the dataset_2d is not existed!!!')

    #==========================================================2、特征分类器============================================================

    if args.model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5,metric='euclidean',weights='distance',algorithm='auto')  #knn分类器
    elif args.model_type == 'SVM':
        model = SVC(C=1.0, gamma=1e-5, kernel='rbf')   # SVM分类器
    elif args.model_type == 'RF':
        model = RandomForestClassifier()  #随机森林分类器
    else:
        print("The classifier is not exsited!")

    model.fit(train_x,train_y)  #training
    y_pred = model.predict(test_x)  #预测结果
    acc = model.score(test_x,test_y)  #准确率
    print('{}/{}({:.3f}%)'.format((y_pred == test_y).astype(int).sum(), test_y.shape[0], acc*100))


    #  ==========================================================3、visualization=====================================================
    if args.visualization == True:
        visualization(args=args,test_y=test_y,y_pred=y_pred,test_x=test_x)

    return acc