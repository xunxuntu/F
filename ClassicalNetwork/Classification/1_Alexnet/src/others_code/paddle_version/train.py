import paddle
import paddle.nn.functional as F
from model import AlexNet


# 读取数据
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(paddle.vision.transforms.Resize(size=resize))  # 将输入数据调整为指定大小
    trans.append(paddle.vision.transforms.ToTensor())
    transform = paddle.vision.transforms.Compose(trans)
    mnist_train = paddle.vision.datasets.FashionMNIST(mode='train',
                                                      transform=transform)  # 由于ImageNet太大所以使用了FashionMNIST数据集
    mnist_test = paddle.vision.datasets.FashionMNIST(mode='test', transform=transform)
    train_iter = paddle.io.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, places=paddle.CPUPlace())
    test_iter = paddle.io.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
                                     places=paddle.CPUPlace())  # places=paddle.CPUPlace()缺少这个关键字会莫名其妙的停止程序，飞桨官网的解释是当前还不支持在子进程中进行GPU Tensor的操作，请不要在子进程流程中使用GPU Tensor
    return train_iter, test_iter


batch_size = 128
# 如出现“out of memory”的报错信息，可减⼩batch_size或resize
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)


# 加载训练集 batch_size 设为 128
def train(model):
    model.train()
    epochs = 5
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_iter()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)  # 交叉熵损失函数
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)  # 计算数据的准确度
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(),
                                                                                acc.numpy()))
            optim.step()
            optim.clear_grad()


model = AlexNet()
train(model)
