import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet
from pathlib import Path

current_file_path = Path(__file__).resolve()  # 获取当前文件的绝对路径
tuc_path = current_file_path.parents[5]  # 获取项目根目录文件路径
datasets_path = tuc_path / "largeFiles" / "datasets"  # 获取数据集路径


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 裁剪到224x224的大小
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # get data root path
    # image_path = os.path.join(data_root, "datasets/flower_data")  # flower data set path
    # print(f'data_root: {data_root}')
    # print(f'image_path: {image_path}')
    image_path = datasets_path / "flower_data"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=image_path / "train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    print(f'train_num: {train_num}')

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())  # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers:开启的线程数
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())  # 查看模型的参数
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 定义一个优化器

    epochs = 5
    model_save_path = tuc_path / "largeFiles" / "model_saved" # 获取模型保存路径
    model_save_path.mkdir(parents=True, exist_ok=True)
    model_path = model_save_path / "AlexNet_2024-04-11.pth"
    best_acc = 0.0  # 最佳准确率
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()  # 这个是用来启动 dropout 方法的
        running_loss = 0.0  # 统计训练中的平均损失
        train_bar = tqdm(train_loader, file=sys.stdout)
        # t1 = time.perf_counter()
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
 
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():  # 在验证中是不需要计算损失梯度的
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), model_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
