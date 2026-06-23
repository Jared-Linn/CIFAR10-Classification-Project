"""
模型 A：Baseline CNN
结构：2层卷积 + 2层全连接
"""
import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    基线卷积神经网络
    - 2 个卷积层 (32, 64 filters)
    - 2 个全连接层 (128, 10)
    - 适合 CIFAR-10 的轻量级模型
    """

    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()

        # 特征提取器
        self.features = nn.Sequential(
            # Block 1: Conv 3x3, 32 filters
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Block 2: Conv 3x3, 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
