"""
class ModelDNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 32, bias=True)

        self.fc2 = nn.Linear(32, 32, bias=True)

        # self.fc3 = nn.Linear(64, 128, bias=True)

        self.out = nn.Linear(32, 1, bias=True)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))

        # x = torch.sigmoid(self.fc3(x))

        x = torch.sigmoid(self.out(x))

        return x


class Train(object):

    def __init__(self):
        self.model = ModelDNN().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)

    def batch_divide(self, size, data, i):
        input_batch = data[i * size:(i + 1) * size, :4]
        output_batch = data[i * size:(i + 1) * size, 4:]
        return input_batch, output_batch

    def training(self, dataset, testset):
        global corr_best

        for i in range(epoch):

            for j in range(0, dataset.shape[0] // batch_size):
                batch_inp, batch_outp = self.batch_divide(batch_size, dataset, j)

                pred = self.model(batch_inp)

                # loss = self.criterion(batch_outp, pred)

                loss = 0.5 * torch.sum((pred - batch_outp)**2)
                # 相对准确率
                acc = torch.abs((pred - batch_outp) / batch_outp)
                acc = acc.cpu().detach().numpy().reshape(-1, ).mean()

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i % 200 == 0) and (j % 1 == 0):
                    print('Train Epoch : {} [{:0>4d}/{}]\tLoss: {:.6f}\tAccurancy: {:.2f}%'.format(i, batch_size * j,
                                                                                                   len(dataset),
                                                                                                   loss.item(),
                                                                                                   (1 - acc) * 100))
                    corr, test_pred = self.testing(testset)

                    if corr < corr_best:
                        corr_best = corr
                        torch.save(self.model, save_path)

        print("Train Epoch:", i)
        axis_x = np.linspace(0, 5e10, f_interval)
        S_test_reshape = S_test.reshape(-1, f_interval)
        test_pred = test_pred.reshape(-1, f_interval)

        S_test_renorm, test_output_renorm = data_renorm(
            S_test_reshape.to('cpu'), test_pred.to('cpu'), tensor_meanY.to('cpu'), tensor_stdY.to('cpu'))
        # print(S_test_renorm - test_y)
        for n in range(5):
            acc_test = torch.abs((test_output_renorm[n, :] - S_test_renorm[n, :]) / S_test_renorm[
                                                                                           n, :])
            print("测试集{}的准确率为:".format(n+1), (1 - acc_test.cpu().detach().numpy().reshape(-1, ).mean()) * 100)
            plt.figure()
            plt.plot(axis_x, S_test_renorm.cpu()[n, :], 'k-', label='actual')
            plt.plot(axis_x, test_output_renorm.cpu().detach().numpy()[n, :],
                     'r-', label='prediction')
            plt.xlabel('f/Hz')
            plt.ylabel('S_Parameter')
            plt.title('No' + str(n + 1))
            plt.legend()

            plt.text(10, -80, "Accuracy:" + str(
                "%.2f" % ((1 - acc_test.cpu().detach().numpy().reshape(-1, ).mean()) * 100)) + "%")
            plt.ylim(-140, 1)
        plt.show()

    def testing(self, testset):
        with torch.no_grad():
            test_inp, test_out = self.batch_divide(testset.shape[0], testset, 0)

            test_pred = self.model(test_inp)

            correct = torch.abs((test_pred - test_out) / test_out)
            correct = correct.cpu().detach().numpy().reshape(-1, ).mean()
            # print('Test set : Accurancy: {:.3f}%'.format((1 - correct) * 100))
            # print("================================================================================================")
        return correct, test_pred

"""