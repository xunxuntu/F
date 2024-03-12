import paddle
import paddle.nn.functional as F


# 组网
class AlexNet(paddle.nn.Layer):
    def __init__(self) -> None:
        super(AlexNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = paddle.nn.Conv2D(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = paddle.nn.Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = paddle.nn.Conv2D(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = paddle.nn.Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool1 = paddle.nn.MaxPool2D(3, 2)
        self.pool2 = paddle.nn.MaxPool2D(3, 2)
        self.pool3 = paddle.nn.MaxPool2D(3, 2)
        self.linear1 = paddle.nn.Linear(in_features=256 * 5 * 5, out_features=4096)
        self.linear2 = paddle.nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = paddle.nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.linear3(x)
        return x
