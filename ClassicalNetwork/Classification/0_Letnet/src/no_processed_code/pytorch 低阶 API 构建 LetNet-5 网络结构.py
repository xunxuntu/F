"""
pytorch 低阶 API 构建 LetNet-5 网络结构
参考链接： https://python.iitter.com/other/174382.html
date:2021.12.24
"""

import torch  # 张量的有关运算，如创建、索引、连接、转置....和numpy的操作很像
import torch.nn as nn  # 八个搭建神经网络层的模块、loss等等
import torch.nn.functional as F  # 常用的激活函数都在这里面
from torchsummary import summary


class LeNet_5(nn.Module):
    """
    模型的定义

    下面这个模型定义没有用Sequential来定义，Sequential的定义方法能够在init中就给出各个层
    之间的关系，我这里是根据是否有可学习的参数。我将可学习参数的层（如全连接、卷积）放在构造函数
    中（其实你想把不具有参数的层放在里面也可以），把不具有学习参数的层(如dropout，
    ReLU等激活函数、BN层)放在forward。
    """

    def __init__(self):
        """
        定义神经网络结构
        """
        # super(LeNet, self).__init__() # python2的写法
        super().__init__()  # python3 的写法

        # 第一个卷积块，这里输入的是单通道，灰色图。
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1))  # stride默认是1，可不用写
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 稠密块，包含三个全连接层
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        正向传播
        x是输入数据，是一个tensor
        """
        x = F.relu(self.conv1(x))  # input(1, 32, 32) output(6, 28, 28)
        x = self.pool1(x)  # output(6, 14, 14)
        x = F.relu(self.conv2(x))  # output(16, 10, 10)
        x = self.pool2(x)  # output(16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)  # output(16*5*5) -1：不确定reshape成几行，但肯定是要reshape成为4列，这时候就用-1
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = F.softmax(self.fc3(x), dim=1)  # output(10)
        return x


net = LeNet_5()
print(net)

print('*' * 2022)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
summary(net, input_size=(1, 32, 32))
