import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 解决多继承中调用父类方法出现的问题，一般继承中都会使用这个super()函数
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 最后输出是10个类别，因为使用的是 cifar10 数据集

    def forward(self, x):
        """
            实现正向传播过程
            N=(W-F+2P)/S+1
        """
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)， 池化层是不改变特征图深度的
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)  输出的时候在内部实现了一个比softmax高级的层
        return x


if __name__ == '__main__':
    input1 = torch.rand([32, 3, 32, 32])
    model = LeNet()
    print(model)
    output = model(input1)
    print(output)
    print(output.shape)
