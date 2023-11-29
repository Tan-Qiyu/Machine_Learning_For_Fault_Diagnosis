'''
可视化函数
'''
import warnings
import datetime
from matplotlib import font_manager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from 图神经网络故障诊断.utils.confusion import confusion

plt.rcParams['font.sans-serif'] = ['SimSun']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def visualization(args,test_y,y_pred,test_x):

    if args.dataset_name == 'SEU':
        if 'bearingset' in args.dataset_path:
            save_name = args.dataset_name + '-bearing-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
        elif 'gearset' in args.dataset_path:
            save_name = args.dataset_name + '-gear-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
    else:
        save_name = args.dataset_name + '-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'

    # 进行混淆矩阵和tsne可视化
    label = test_y  # the label of testing dataset_2d
    prediction = y_pred  # 预测标签
    if args.dataset_name == 'CWRU' or args.dataset_name == 'SEU' or args.dataset_name == 'DC':
        classes = list(range(10))
    elif args.dataset_name == 'XJTU' or args.dataset_name == 'MFPT':
        classes = list(range(15))
    elif args.dataset_name == 'Bearing_311' or args.dataset_name == 'UoC':
        classes = list(range(9))
    elif args.dataset_name == 'QianPeng_gear':
        classes = list(range(6))
    elif args.dataset_name == 'JNU':
        classes = list(range(12))
    elif args.dataset_name == 'Wind_Gearbox':
        classes = list(range(args.wind_mode))
    elif args.dataset_name == 'SQV':
        classes = list(range(7))

    confusion_data = confusion_matrix(label, prediction)

    confusion(confusion_matrix=confusion_data)  #绘制混淆矩阵

    if args.save_visualization_results == True:
        plt.savefig(args.visualization_dirpath + '\混淆矩阵-{}.tif'.format(save_name), dpi=300, bbox_inches='tight')  # 保存混淆矩阵图

    # tsne visualization
    def plot_embedding(data, label):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure(figsize=(5,4))

        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内`

        ax = plt.subplot(111)
        # 创建局部变量存放Plot以绘制相应的legend
        my_font = font_manager.FontProperties(family='Times New Roman', size=8)

        fig_leg = []
        list(map(lambda x: fig_leg.append(x), range(len(classes))))
        marker = ['o', '^', 'p', 'P', '*', 's', 'x', 'X', '+', 'd', 'D', '>', 'H', 'h', '<', '1', '2']
        for i in range(data.shape[0]):
            fig_leg[int(label[i])] = plt.plot(data[i, 0], data[i, 1], linestyle='', marker=marker[label[i]],
                                              markersize=9,color=plt.cm.tab10(label[i] / 10.) if len(classes) <= 10 else plt.cm.tab20(label[i] / 20.))
        hand = list(map(lambda x: x[0], fig_leg))
        plt.legend(loc='right', ncol=1, frameon=True, labelspacing=0.8, columnspacing=0.4, handletextpad=0.4,
                   prop=my_font, handlelength=1, bbox_to_anchor=(1.1, 0.5),
                   handles=hand, labels=classes)

        # plt.xticks(fontproperties=font_manager.FontProperties(family='Times New Roman', size=8))
        # plt.yticks(fontproperties=font_manager.FontProperties(family='Times New Roman', size=8))
        plt.xticks([])
        plt.yticks([])
        plt.xlim([data[:, 0].min() - 0.05, data[:, 0].max() + 0.05])
        plt.ylim([data[:, 1].min() - 0.05, data[:, 1].max() + 0.05])

        if args.save_visualization_results == True:
            plt.savefig(args.visualization_dirpath + '\时频域特征可视化-{}.tif'.format(save_name), dpi=300,bbox_inches='tight')  # 保存tsne可视化结果

        return fig

    warnings.filterwarnings('ignore')
    tsne = TSNE(n_components=2, init='pca')

    result = tsne.fit_transform(test_x)  # 对特征进行降维
    fig = plot_embedding(result, label)

    plt.show()