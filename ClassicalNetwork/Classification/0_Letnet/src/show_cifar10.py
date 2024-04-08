"""
查看cifar10 数据集
    在mobaxterm中可以正常查看
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 串联多个图片的变换操作
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 标准化:output=(input-mean)/std

# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='../../datasets',
                                        train=False,  # 表示是数据集中的测试集
                                        download=False, transform=transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=4,  # 每批用于验证的样本数
                                          shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)  # 迭代器
test_image, test_label = next(test_data_iter)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize 反标准化 input=output*0.5+0.5=output/2+0.5
    npimg = img.numpy()
    # 调用ToTensor()方法会自动将输入数据变为[batch, channel, height, wight]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 将pytorch的tensor通道顺序是[channel, height, wight]转化为[height, wight, channel]
    plt.show()

# print labels
print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# show images
imshow(torchvision.utils.make_grid(test_image))

