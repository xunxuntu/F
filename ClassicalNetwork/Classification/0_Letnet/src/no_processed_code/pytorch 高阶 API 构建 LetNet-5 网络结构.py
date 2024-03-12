"""
pytorch 高阶 API 构建 LetNet-5 网络结构
参考链接：https://huhuhang.com/post/machine-learning/lenet-5
date:2021.12.24
"""

import torch.nn as nn


class Flatten(nn.Module):
    """
    PyTorch 也提供了 nn.Sequential 容器结构，类似于 Keras 的使用过程。
    nn.Sequential 可以简化模型的构建过程，直接使用 torch.nn 提供的类来组合即可。
    注意，PyTorch 中未提供 Flatten 类，所以需要使用 reshape 操作预先定义一个。
    """
    def forward(self, input):
        return input.reshape(input.size(0), -1)


# 构建 Sequential 容器结构
model = nn.Sequential(
    nn.Conv2d(1, 6, (5, 5)),
    nn.ReLU(),
    nn.AvgPool2d((2, 2)),
    nn.Conv2d(6, 16, (5, 5)),
    nn.ReLU(),
    nn.AvgPool2d((2, 2)),
    Flatten(),
    nn.Linear(5 * 5 * 16, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10),
    nn.Softmax(dim=1)
)
print(model)
