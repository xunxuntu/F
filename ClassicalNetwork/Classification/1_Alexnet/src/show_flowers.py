"""
    需要将batsize设为4, shuffle设为True才可以正常显示, 如果设置的比较大的话就显示不正常了。
"""
import os
import torch
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    plt.imsave('my_image.jpg', img)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # get data root path
    image_path = os.path.join(data_root, "my_datasets/FlowerPhotos")  # flower data set path
    print(data_root)
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # print(train_num)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())  # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 查看数据集图片代码
    # 需要将 validate_loader 中的 batch_size 设置为一个小的数字，比如2,3,4,1，...，大了不好显示
    test_data_iter = iter(validate_loader)
    test_image, test_label = next(test_data_iter)
    
    print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    imshow(utils.make_grid(test_image))


if __name__ == '__main__':
    main()
