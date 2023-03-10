import torch
import numpy as np
import pandas as pd
import scipy.signal as signal
import math


def load_physical_para_data(file_path):
    """
        加载物理参数，即输入数据的CSV文件
    :param file_path: 文件的路径
    :return: var_idx为索引值，inp_data具体输入变量【物理参数长度】
    """
    df = pd.read_csv(file_path, header=None)
    df_data = df.values
    # var_idx = df_data[:, 0].astype(int)  # 索引
    # inp_data = df_data[:, 0:]
    inp_data = df_data[0:, :]
    return inp_data


def load_freq_data(file_path):
    """
        根据var_idx读取S参数文件    .npy
    :param file_path: S参数文件夹路径
    :param file_name_head: .npy文件统一名字的前缀
    :param var_idx: 输入数据的索引
    :return: 返回读取的对应的数据集，ndarray形式
    """
    df = pd.read_csv(file_path, header=None)
    df_data = df.values
    # var_idx = df_data[:, 0].astype(int)  # 索引
    # inp_data = df_data[:, 0:]
    inp_data = df_data
    inp_data = inp_data[:, 1:]
    """
    # 对除第0列外的数据去除单位mil
    for i in range(inp_data.shape[0]):
        for j in range(inp_data.shape[1]):
            # inp_data[i, j] = inp_data[i, j][:-3]
            inp_data[i, j] = inp_data[i, j]
    inp_data = inp_data.astype(float)
    # return var_idx, inp_data
    """

    return inp_data


def divede_dataset(data_x, data_y, train_size):
    """
        训练集和测试集的划分
    :param data_x: 数据
    :param data_y: 标签
    :return: 训练集数据，训练集标签，测试集数据，测试集标签
    """
    # 乱序
    m = data_x.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_x, shuffled_y = data_x[permutation, :], data_y[permutation, :]
    # shuffled_x, shuffled_y = data_x, data_y
    # test_size = 200
    # train_size = 648
    train_x, train_y = shuffled_x[:train_size, :], shuffled_y[:train_size, :]
    test_x, test_y = shuffled_x[train_size:, :], shuffled_y[train_size:, :]
    return train_x, train_y, test_x, test_y


def data_pre_process(train_x_data, train_y_data, test_x_data, test_y_data):
    """
    对数据进行预处理: 归一化，转成Tensor
    :param train_x_data: 训练集数据
    :param train_y_data: 训练集标签
    :param test_x_data:  测试集数据
    :param test_y_data:  测试集标签
    :return: tensor
    """

    # y坐标幅值提前对数化
    # train_y_data = 20 * np.log10(train_y_data)
    # test_y_data = 20 * np.log10(test_y_data)

    # 输入的特征进行归一化，标签进行标准化
    # Z-score标准化输出y，min-max归一化输入x
    # 需获得均值和标准差，所以用训练集的来替代，因为测试集的是不定的，测试集标准化便可以用这个
    meanY = train_y_data.mean(axis=0).reshape(1, -1)  # 在0维中的均值,求每一列的均值
    stdY = train_y_data.std(axis=0).reshape(1, -1)  # 在0维中的标准差,求每一列的标准差

    # mean_test_Y = test_y_data.mean(axis=1).reshape(-1, 1)
    # test_stdY = test_y_data.std(axis=1).reshape(-1, 1)
    # Inputs are scaled between [-1, 1].
    meanX = train_x_data.min(axis=0).reshape(1, -1)
    stdX = (train_x_data.max(axis=0) - train_x_data.min(axis=0)).reshape(1, -1)

    # Normalize Train and Test Data
    # Use mean and std of train data to normalize test data to avoid bias
    # 当stdx值为0时归一化值为-1，避免nan产生

    training_x = -1 + 2 * np.divide((train_x_data - meanX), stdX, out=np.zeros_like(train_x_data), where=stdX != 0)

    training_y = (train_y_data - meanY) / stdY  # 训练集标签

    test_x = -1 + 2 * np.divide((test_x_data - meanX), stdX, out=np.zeros_like(test_x_data), where=stdX != 0)

    test_y = (test_y_data - meanY) / stdY   # 测试集标签

    # Convert everything to tensor
    tensor_x = torch.Tensor(training_x.astype(float))  # 训练集数据
    tensor_y = torch.Tensor(training_y.astype(float))  # 训练集标签
    tensor_test_x = torch.Tensor(test_x.astype(float))  # 测试集数据
    tensor_test_y = torch.Tensor(test_y.astype(float))  # 测试集标签

    tensor_meanX = torch.Tensor(meanX.astype(float))
    tensor_stdX = torch.Tensor(stdX.astype(float))
    tensor_meanY = torch.Tensor(meanY.astype(float))
    tensor_stdY = torch.Tensor(stdY.astype(float))
    # tensor_mean_test_Y = torch.Tensor(mean_test_Y)
    # tensor_test_stdY = torch.Tensor(test_stdY)
    return tensor_x, tensor_y, tensor_test_x, tensor_test_y, tensor_meanY, tensor_stdY, tensor_meanX, tensor_stdX


def data_build(tensor_x, tensor_y, f_interval, f_parameter):
    # 先生成tensor用来存储操作后的数组
    train = torch.zeros((tensor_x.shape[0] * f_interval, 3))  # 所有几何特征
    train_data_all = torch.zeros((tensor_x.shape[0] * f_interval, 4))  # 几何特征和频率特征
    tensor_y_norm = torch.zeros((tensor_y.shape[0] * f_interval, 1))
    # tensor_yy = torch.zeros((500, 1))
    for i in range(tensor_x.shape[0]):  # 选择训练集开始的个数
        train[i * f_interval: (i + 1) * f_interval, :] = tensor_x[i, :].reshape(-1, 3).repeat(
            f_interval, 1)  # 将样本的几何特征复制
        train_data_all[i * f_interval:(i + 1) * f_interval, :] = torch.cat(
            (train[i * f_interval:(i + 1) * f_interval, :].reshape(-1, 3),
             f_parameter.reshape(-1, 1)), dim=1)
        tensor_yy = tensor_y[i, :].reshape(50, 1)
        tensor_y_norm[i * f_interval: (i + 1) * f_interval, :] = tensor_yy[0:50:(50 // f_interval),
                                                                 0].reshape(f_interval, 1)
    tensor_y_norm = tensor_y_norm.reshape(train_data_all.shape[0], 1)
    data_train = torch.cat((train_data_all, tensor_y_norm), dim=1)

    return data_train


def data_renorm(test_x, test_y, mean_y, std_y):
    """
    test_x是测试集实际的标签
    test_y是测试集预测得到的标签
    test_x和test_y的维度都是test_number*f_interval
    """
    std_test_x = torch.std(test_x, dim=0).reshape(1, -1)
    std_test_y = torch.std(test_y, dim=0).reshape(1, -1)

    mean_test_x = torch.mean(test_x, dim=0).reshape(1, -1)
    mean_test_y = torch.mean(test_y, dim=0).reshape(1, -1)

    test_x = test_x * std_y + mean_y
    test_y = test_y * std_y + mean_y
    return test_x, test_y


def data_x_norm(x, mean_x, std_x):
    """
        对待预测的输入电容向量数据进行归一化
        归一化的区间值为【-1，1】
    """
    test_x = -1 + 2 * np.divide((x - mean_x), std_x, out=np.zeros_like(x), where=std_x != 0)
    tensor_test_x = torch.Tensor(test_x)

    return tensor_test_x


def data_y_norm(y, mean_y, std_y):
    """
        对待预测的输出阻抗曲线数据进行归一化
    """
    y = 20 * np.log10(y)
    norm_y = (y - mean_y) / std_y
    norm_y = torch.Tensor(norm_y)
    return norm_y


def data_y_denorm(y, mean_y, std_y):
    """
        对输出阻抗曲线数据进行反归一化
    """
    raw_y = (y * std_y + mean_y).cpu()
    # raw_y = torch.pow(10, raw_y/20)   # 画图需要对数化纵轴，反归一化计算准确率便不再操作
    return raw_y


def data_x_denorm(x, mean_x, std_x):
    """
        对输入电容向量进行反归一化
    """
    raw_x = ((x + 1) * std_x / 2 + mean_x).cpu()
    return raw_x


def get_y_extremum(inp, freq_array):
    """
        获取阻抗曲线的极值点索引
    """
    y_extremum = []  # 测试集每个曲线的极值点索引列表
    for i in range(inp.shape[0]):
        test_extremum_max = signal.argrelextrema(np.array(inp[i, :]), np.greater)  # 极大值
        test_extremum_min = signal.argrelextrema(np.array(inp[i, :]), np.less)  # 极小值
        test_extremum_max = np.array(test_extremum_max).flatten()
        test_extremum_min = np.array(test_extremum_min).flatten()
        temp_extremum = np.sort(np.concatenate((test_extremum_max, test_extremum_min)))
        # for j in range(temp_extremum.shape[0]):
        #     # 只取前三部分的极值点，大约在5e8频点周围
        #     if freq_array[temp_extremum[j]] > 6e8:
        #         temp_extremum = temp_extremum[:j]
        #         break
        temp_extremum = list(temp_extremum)
        temp_extremum.insert(0, 0)  # 添加起点和末点
        temp_extremum.append(430)
        y_extremum.append(temp_extremum)
    return y_extremum


if __name__ == "__main__":
    print()
