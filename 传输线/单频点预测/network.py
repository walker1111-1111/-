import torch
from torch import nn
import numpy as np
from data_process import *
import matplotlib.pyplot as plt
import time

# ===============Parameters and Flags===============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
save_path = "./dnn_model/model_file.pth"
save_img_path = "./img"
# =============================================
corr_best = float("inf")

# ===============File adress===============
data_physical_name = './data/Tlines_2.csv'
input_size_geometry = 3  # 第一部分的输入特征数
output_size_first = 32  # 第一部分的输出维数
output_size_all = 1  # 第二部分的输出维数
input_size_second = output_size_first + 1  # 第二部分的输入维数
# ===============Hyper parameter===============
epoch = 2000
f_interval = 50  # 预测频率点的个数
sample_number = 700  # 训练集个数为700
test_number = 720 - sample_number
batch_size = sample_number * f_interval


class ModelDNN_part1(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(input_size_geometry, 64)

        self.fc2 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, output_size_first)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))

        # x = F.sigmoid(self.fc3(x))

        out = torch.sigmoid(self.fc4(x))

        return out


class ModelDNN_part2(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(input_size_second, 128, bias=True)

        self.fc2 = nn.Linear(128, 64)

        self.out = nn.Linear(64, output_size_all, bias=True)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))

        # x = torch.sigmoid(self.fc3(x))

        return self.out(x)


class Train(object):

    def __init__(self):
        self.model_first = ModelDNN_part1().to(device)  # 执行Forward
        self.criterion = nn.MSELoss()
        self.optimizer_first = torch.optim.Adam(self.model_first.parameters(), lr=0.02)

        self.model_second = ModelDNN_part2().to(device)  # 执行Forward
        self.criterion = nn.MSELoss()
        self.optimizer_second = torch.optim.Adam(self.model_second.parameters(), lr=0.02)

    def batch_divide(self, size, data, i):
        input_batch = data[i * size:(i + 1) * size, 0:input_size_geometry]  # 取出物理特征
        input_batch_f = data[i * size:(i + 1) * size, input_size_geometry]  # 取出频率特征
        output_batch = data[i * size:(i + 1) * size, -1]  # 取出标签
        return input_batch, input_batch_f, output_batch

    def training(self, dataset, testset):
        global corr_best
        start = time.time()
        for i in range(epoch):

            for j in range(0, dataset.shape[0] // batch_size):
                batch_inp, batch_f, batch_outp = self.batch_divide(batch_size, dataset, j)

                output_first = self.model_first(batch_inp[:, 0:input_size_geometry])

                input_second = torch.cat((output_first, batch_f[:].reshape(batch_f.shape[0], -1)),
                                         dim=1)  # 将第一部分的输出与频率参数相结合，送入第二部分

                output_second = self.model_second(input_second)
                batch_outp = batch_outp.reshape(batch_outp.shape[0], -1)

                loss_second = 0.5 * torch.sum((output_second - batch_outp) ** 2)
                # 反向传播
                self.optimizer_first.zero_grad()
                self.optimizer_second.zero_grad()

                loss_second.backward()
                self.optimizer_first.step()
                self.optimizer_second.step()

                # 相对准确率

                acc = torch.abs((output_second - batch_outp) / batch_outp)

                acc = acc.cpu().detach().numpy().reshape(-1, ).mean()

                test_output_second, corr = self.testing(testset)

                if corr < corr_best:
                    corr_best = corr
                    torch.save(self.model_second, save_path)

                if (i % 200 == 0) and (j % 20 == 0):
                    print('Train Epoch : {} [{:0>4d}/{}]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(i, batch_size * j,
                                                                                                  len(dataset),
                                                                                                  loss_second.item(),
                                                                                                  (1 - acc) * 100))

        print("模型训练完毕")
        end = time.time()
        print("训练时间为:", end - start)
        axis_x = np.linspace(0, 1e10, f_interval)
        S_test_reshape = S_test.reshape(test_number, f_interval)
        test_output_second_reshape = test_output_second.reshape(test_number, f_interval)

        S_test_renorm, test_output_second_renorm = data_renorm(
            S_test_reshape.to('cpu'), test_output_second_reshape.to('cpu'), tensor_meanY.to('cpu'),
            tensor_stdY.to('cpu'))

        for n in range(test_number):
            acc_test = torch.abs((test_output_second_renorm[n, :] - S_test_renorm[n, :]) / S_test_renorm[
                                                                                           n, :])

            print("测试集样本" + str(n + 1) + "的准确率为：",
                  (1 - acc_test.cpu().detach().numpy().reshape(-1, ).mean()) * 100)
            plt.figure()
            plt.plot(axis_x, S_test_renorm.cpu()[n, :], 'k-', label='actual')
            plt.plot(axis_x, test_output_second_renorm.cpu().detach().numpy()[n, :],
                     'r-', label='prediction')
            plt.xlabel('f/Hz')
            plt.ylabel('S_Parameter')
            plt.title('No' + str(n + 1))
            plt.legend()

            plt.text(10, -80, "Accuracy:" + str(
                "%.2f" % ((1 - acc_test.cpu().detach().numpy().reshape(-1, ).mean()) * 100)) + "%")

            plt.ylim(-140, -10)
            plt.savefig(save_img_path + 'No.' + str(n+1) + '.jpg')  # 保存图片

    def testing(self, testset):
        with torch.no_grad():
            test_inp, test_f, test_out = self.batch_divide(testset.shape[0], testset, 0)
            test_input_first = self.model_first(test_inp[:, 0:input_size_geometry])
            test_input_second = torch.cat(
                (test_input_first, testset[:, input_size_geometry].reshape(testset.shape[0], -1)),
                dim=1)  # 将第一部分的输出与频率参数相结合，送入第二部分
            test_output_second = self.model_second(test_input_second)
            correct = torch.abs((test_output_second.to(device) - S_test) / S_test)
            correct = correct.cpu().detach().numpy().reshape(-1, ).mean()
            # print('Test set : Accuracy: {:.2f}%'.format((1 - correct) * 100))
            # print("================================================================================================")
        return test_output_second, correct


if __name__ == '__main__':
    # 载入数据， 将频率参数作为输入参数
    data_all = load_physical_para_data(data_physical_name)  # 读取所有矩阵参数

    physical_parameter = data_all[:, :3]

    physical_parameter = physical_parameter[::51, :].astype(np.float64)  # 取出其中一组物理参数
    output_parameter = data_all[:, -1].reshape(-1, 51)  # 得到所有情况下所有频率的输出

    output_parameter = output_parameter[:, 1:].astype(np.float64)  # 取后面的50个输出，因为在0Hz情况下的输出都一样
    # 划分训练集和测试集，训练集选择720个
    train_x, train_y, test_x, test_y = divede_dataset(physical_parameter, output_parameter, sample_number)
    # 将训练集合测试集的数据标准化
    tensor_x, tensor_y, tensor_test_x, tensor_test_y, tensor_meanY, tensor_stdY, tensor_meanx, tensor_stdx = \
        data_pre_process(train_x, train_y, test_x, test_y)
    # 由于本次实验采用的是单频点预测，所以需要将几何特征每行复制50次
    # 先将频率复制50次
    f_parameter = -1 + 2 * torch.linspace(0, 5e10, 51) / 5e10
    f_parameter = f_parameter[1:]
    train_data = data_build(tensor_x, tensor_y, f_interval, f_parameter)  # 生成训练数据集

    test_data = data_build(tensor_test_x, tensor_test_y, f_interval, f_parameter)  # 生成测试数据集
    S_train = train_data[:, -1].reshape(-1, 1).to(device)
    S_test = test_data[:, -1].reshape(-1, 1).to(device)

    model = Train()
    model.training(train_data.to(device), test_data.to(device))
