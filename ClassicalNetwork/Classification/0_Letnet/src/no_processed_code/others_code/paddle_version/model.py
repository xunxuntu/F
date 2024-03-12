import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as f


#  定义LeNet网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num):  # num:分类标签的分类数
        super(LeNet, self).__init__()
        #  创建第一个卷积和池化层
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第二个卷积和池化层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        #  创建第三个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 创建全连接层，第一个的输出神经元个数为64， 第二个的输出神经元个数为分类标签的类别数
        self.fc1 = Linear(in_features=120, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num)

    # 卷积神经网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        #  每个卷积层使用relu函数，而后是一个2*2的池化
        x = f.relu(x)
        x = self.max_pool1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸逻辑：输入层将数据拉平[B,C,H,W] ->[B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        return x
