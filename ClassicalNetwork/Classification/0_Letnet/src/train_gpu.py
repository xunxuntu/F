"""
参考链接:https://blog.csdn.net/m0_37867091/article/details/107136477
date:2021.12.27
"""

import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 串联多个图片的变换操作
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 标准化:output=(input-mean)/std

# 导入50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='../../../../../Datasets/CIFAR10',  # 数据集存放目录
                                         train=True,  # 表示导入数据集中的训练集
                                         download=True,  # 第一次运行时为True，下载数据集，下载完成后改为False
                                         transform=transform)  # 图片预处理过程
# torchvision.datasets. 后面有很多的数据集可以下载

# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set,  # 导入的训练集
                                           batch_size=36,  # 每批训练的样本数，每一批拿出36张图片进行训练
                                           shuffle=True,  # 是否打乱训练集
                                           num_workers=0)  # 载入数据使用的线程数，在windows下只能设置为0

# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='../../../../../Datasets/CIFAR10',
                                        train=False,  # 表示是数据集中的测试集
                                        download=False, transform=transform)

# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=10000,  # 每批用于验证的样本数
                                          shuffle=False, num_workers=0)
# print(type(test_loader))

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)  # 迭代器
test_image, test_label = next(test_data_iter)
# test_image, test_label = test_data_iter.next()  # 原版本的写法

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()  # 定义训练的网络模型
net.to(device)  # 将网络分配到指定的device中
# lenet最后一层之所以没有softmax这个函数，是因为CrossEntropyLoss包含了softmax
loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数

optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）111

for epoch in range(50):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0  # 训练中的损失
    time_start = time.perf_counter()  # 返回当前的计算机系统时间

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算

        inputs, labels = data  # 获取训练集的图像和标签
        optimizer.zero_grad()  # 清除历史梯度 ,原因：https://www.zhihu.com/question/303070254

        # forward + backward + optimize
        outputs = net(inputs.to(device))  # 正向传播
        loss = loss_function(outputs, labels.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        # 打印耗时、损失、准确率等数据
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batches，每500步打印一次
            with torch.no_grad():  # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image.to(device))  # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                      (epoch + 1, step + 1, running_loss / 500, accuracy))

                print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
                
                running_loss = 0.0

print('Finished Training')

# 保存训练得到的参数
save_path = '../workspace/Lenet_20240313.pth'
torch.save(net.state_dict(), save_path)
