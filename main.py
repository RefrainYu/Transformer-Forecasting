"""
erditor: Snu77
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from util.data_factory import data_provider
from util.tools import adjust_learning_rate
# params
from layers import Transformer
import matplotlib.pyplot as plt

# 搭建模型Transoformer模型
class Transformerinitialization():
    def __init__(self, args):
        super(Transformerinitialization, self).__init__()
        self.args = args
        self.model, self.device = self.build_model(args)
        # 个人添加：初始化训练损失列表
        self.train_loss_list = []

    def build_model(self, args):
        model = Transformer.Model(self.args).float()
        # 将模型定义在GPU上

        if args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.device))
            print('Use GPU: cuda:{}'.format(args.device))
        else:
            print('Use CPU')
            device = torch.device('cpu')
        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))  # 打印模型参数量

        if self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=[device])

        return model, device

    def _get_data(self, flag, pre_data=None):
        data_set, data_loader = data_provider(self.args, flag, pre_data)
        return data_set, data_loader

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # 个人添加：收集每个epoch的平均损失
            self.train_loss_list.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'model.pth'
        torch.save(self.model.state_dict(), best_model_path)
        # 个人添加：训练结束后，绘制并保存损失曲线 args.train_epochs
        loss_save_path = path + '/' + 'loss_curve.png'
        self._plot_and_save_loss_curve(loss_save_path)

        return self.model

    # 个人添加
    def _plot_and_save_loss_curve(self, save_path):
        """绘制并保存损失曲线"""
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.train_loss_list) + 1)
        plt.plot(epochs, self.train_loss_list, label='Training Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)  # 保存图片
        plt.close()  # 关闭图表，释放资源

    def calculate_mse(self, y_true, y_pred):
        # 均方误差
        mse = np.mean(np.abs(y_true - y_pred))
        return mse

    def predict(self, setting, load=False):
        # predict
        results = []
        preds = []

        # 加载模型
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'model.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 评估模式
        self.model.eval()

        if args.rollingforecast:
            pre_data = pd.read_csv(args.root_path + args.rolling_data_path)
        else:
            pre_data = None

        for i in 0 if pre_data is None else range(int(len(pre_data)/args.pred_len) - 1):
            if i == 0:
                data_set, pred_loader = self._get_data(flag='pred')
            else:  # 进行滚动预测

                data_set, pred_loader = self._get_data(flag='pred', pre_data=pre_data.iloc[: i * args.pred_len])
                print(f'滚动预测第{i + 1} 次')

            with torch.no_grad():
                for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                        batch_y.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    outputs = data_set.inverse_transform(outputs)

                    if self.args.features == 'MS':
                        for i in range(args.pred_len):
                            preds.append(outputs[0][i][outputs.shape[2] - 1])  # 取最后一个预测值即对应target列
                    else:
                        for i in range(args.pred_len):
                            preds.append(outputs[0][i][outputs.shape[2] - 1])
                    print(outputs)

        # 保存结果
        if args.rollingforecast:
            df = pd.DataFrame({'real': pre_data['{}'.format(args.target)][:len(preds)], 'forecast': preds})
            df.to_csv('./results/{}-ForecastResults.csv'.format(args.target), index=False)
        else:
            df = pd.DataFrame({'forecast': results})
            df.to_csv('./results/{}-ForecastResults.csv'.format(args.target), index=False)

        if args.show_results:

            # 设置绘图风格
            plt.style.use('ggplot')

            # 创建折线图
            plt.plot(df['real'].tolist(), label='real', color='blue')  # 实际值
            plt.plot(df['forecast'].tolist(), label='forecast', color='red')  # 预测值

            # 增强视觉效果
            plt.grid(True)
            plt.title('real vs forecast')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.legend()

            plt.savefig('results.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Multivariate Time Series Forecasting')
    # basic config
    parser.add_argument('--train', type=bool, default=True, help='Whether to conduct training')
    parser.add_argument('--rollingforecast', type=bool, default=True, help='rolling forecast True or False')
    parser.add_argument('--rolling_data_path', type=str, default='1A-Trina-To-Predict-2.csv', help='rolling data file')
    parser.add_argument('--show_results', type=bool, default=True, help='Whether show forecast and real results graph')
    parser.add_argument('--model', type=str, default='Transformer',help='Model name')

    # data loader
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='1A-Real-Trina-new.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Active_Power', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./models/', help='location of model models')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=126, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=64, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')

    # model
    parser.add_argument('--norm', action='store_false', default=True, help='whether to apply LayerNorm')
    parser.add_argument('--rev', action='store_true', default=True, help='whether to apply RevIN')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=12, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=12, help='output size')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + positional embedding')
    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--device', type=int, default=0, help='gpu')
    args = parser.parse_args()
    Exp = Transformerinitialization
    # setting record of experiments
    setting = 'predict-{}-data-{}'.format(args.model, args.data_path[:-4])

    SCI = Transformerinitialization(args)  # 实例化模型
    if args.train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model))
        SCI.train(setting)
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model))
    SCI.predict(setting, True)

    plt.show()