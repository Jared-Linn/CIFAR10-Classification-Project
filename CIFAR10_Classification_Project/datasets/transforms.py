"""
数据增强与预处理 transforms
"""
import torchvision.transforms as transforms
from config.config import CIFAR10_MEAN, CIFAR10_STD


def get_train_transforms():
    """
    训练集数据增强与预处理
    包含：随机裁剪、随机水平翻转、随机旋转、归一化
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_val_transforms():
    """
    验证集/测试集预处理（不做数据增强）
    """
    return transforms.Compose([
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_test_transforms():
    """测试集预处理（同验证集）"""
    return get_val_transforms()
