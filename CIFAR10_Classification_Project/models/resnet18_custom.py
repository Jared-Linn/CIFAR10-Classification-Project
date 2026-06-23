"""
模型 B：改进模型 - 自定义 ResNet-18（适配 CIFAR-10）
改进点：
1. Batch Normalization - 加速收敛、提升稳定性
2. Residual Connection - 缓解梯度消失、提高准确率
3. Dropout - 减少过拟合
"""
import torch
import torch.nn as nn
from models.blocks import ResidualBlock


class ResNet18CIFAR(nn.Module):
    """
    为 CIFAR-10 定制的 ResNet-18
    - 输入尺寸 32x32（原始 ResNet-18 针对 224x224 设计）
    - 移除了第一个 MaxPool 以保留空间信息
    - 使用 BasicBlock 残差单元
    """

    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.5):
        super().__init__()

        self.in_channels = 64

        # 初始卷积层（无 MaxPool，适配 32x32 小图）
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 残差层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 全局平均池化 + 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * ResidualBlock.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        """构建一个残差层"""
        layers = []
        # 第一个 block 可能需要 stride=2 降采样
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * ResidualBlock.expansion
        # 后续 block
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
