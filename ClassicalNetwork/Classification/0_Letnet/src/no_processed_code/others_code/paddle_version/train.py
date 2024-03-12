from model import LeNet
import os
import random
import paddle
import numpy as np
import paddle.nn.functional as f
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST


def train(model, opt, train_loader, valid_loader, epochs):
    #  开启CPU训练
    paddle.device.set_device('cpu')
    print('开始训练...')
    model.train()
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            img = data[0]
            label = data[1]  # 真实值
            logits = model(img)  # 预测值
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            aver_loss = paddle.mean(loss)

            if batch_id % 2000 == 0:
                print(f"epoch: {epoch}, batch_id: {batch_id}, loss is: {float(aver_loss.numpy())}")
            aver_loss.backward()  # 反向传播计算
            opt.step()  # 更新x
            opt.clear_grad()  # 清除梯度
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader):
            img, label = data[0], data[1]  # label:真实值
            logits = model(img)  # 传入神经网络的预测值
            pred = f.relu(logits)  # 直接传入relu函数得到的预测值
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print(f"[validation] accuracy/loss: {np.mean(accuracies)}/{np.mean(losses)}")  # 准确率与平均损失的比值
        model.train()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')


# 创建数据
model = LeNet(num=10)
# 设置迭代次数
epochs = 5
# 设置优化器Momentum, 学习率为0.001
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
# 定义数据读取器
train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=10, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)
# 开始训练
train(model, opt, train_loader, valid_loader, epochs)
