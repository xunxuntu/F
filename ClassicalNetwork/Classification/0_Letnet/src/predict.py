import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         # 训练中使用了标准化处理，验证的时候就也需要标准化处理
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('../workspace/Lenet_20230905.pth'))

    im = Image.open('../workspace/images/feiji.png').convert("RGB")  # 如果输入图像有一个 alpha 通道的像素，那么它是四通道而不是是三通道的图像，需要转为三通道
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W] 增加一个新的维度

    with torch.no_grad():  # 不需要求损失梯度
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
        predict_percent = torch.softmax(outputs, dim=1)  # softmax是二分类函数sigmoid在多分类上的推广,目的是将多分类的结果以概率的形式展现出来
    print(classes[int(predict)])
    print(predict_percent)


if __name__ == '__main__':
    main()