import torch
import torch.nn as nn
from torchvision import models


class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedResNet, self).__init__()
        # 加载预训练的ResNet模型，例如ResNet18
        self.resnet = models.resnet18(pretrained=True)
        # 修改第一个卷积层以接受2个通道
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 修改最后的全连接层以输出2个类别
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        # num_ftrs = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
                # 获取全连接层的原始输出（logits）
        x = self.resnet(x)
        # 应用sigmoid函数将输出转换为概率
        x = torch.sigmoid(x)
        return x
