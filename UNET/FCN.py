import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.conv5(x)  # 最后一层不需要ReLU激活函数，用于输出分割结果
        return x

if __name__ == "__main__":
    # 输入通道数（针对RGB图像为3），输出类别数（分割任务的类别数）
    in_channels = 3
    num_classes = 2  # 例如在Pascal VOC数据集中，有20个物体类别和1个背景类别

    # 创建模型实例
    model = FCN(in_channels, num_classes)

    x = torch.randn((2, 3, 512, 512))
    f = model
    y = f(x)
    print(y.shape)
