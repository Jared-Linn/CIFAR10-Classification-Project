"""
CIFAR-10 数据集加载模块
从原始 CIFAR-10 二进制文件加载数据，支持训练/验证/测试划分
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from config.config import RAW_DATA_DIR, NUM_CLASSES, TRAIN_RATIO, VAL_RATIO, SEED


def unpickle(file_path):
    """读取 CIFAR-10 二进制 batch 文件"""
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
    return data_dict


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 自定义 Dataset，从原始 batch 文件加载
    """

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: numpy array (N, 3, 32, 32) or (N, 3072)
            labels: numpy array (N,)
            transform: torchvision transforms
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # CIFAR-10 原始数据 shape: (3072,) -> (3, 32, 32)
        if img.ndim == 1:
            img = img.reshape(3, 32, 32)

        # 转换为 float 并归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # 转换为 torch tensor (C, H, W)
        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def load_cifar10_data(data_dir=RAW_DATA_DIR):
    """
    加载 CIFAR-10 全部数据

    Returns:
        train_data: (50000, 3072) numpy array
        train_labels: (50000,) numpy array
        test_data: (10000, 3072) numpy array
        test_labels: (10000,) numpy array
    """
    # 加载训练数据 (5个batch)
    train_data_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        batch = unpickle(batch_file)
        train_data_list.append(batch[b"data"])
        train_labels_list.append(batch[b"labels"])

    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.array(train_labels_list).flatten()

    # 加载测试数据
    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    test_data = test_batch[b"data"]
    test_labels = np.array(test_batch[b"labels"])

    return train_data, train_labels, test_data, test_labels


def create_data_splits(train_data, train_labels, transform_train=None, transform_val=None):
    """
    划分训练集、验证集，同时保留测试集

    Args:
        train_data: 原始训练数据 (50000, ...)
        train_labels: 原始训练标签 (50000,)
        transform_train: 训练集 transform
        transform_val: 验证集 transform

    Returns:
        train_dataset, val_dataset
    """
    # 使用分层抽样
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=SEED
    )

    train_indices, val_indices = next(sss.split(train_data, train_labels))

    train_dataset = CIFAR10Dataset(
        train_data[train_indices],
        train_labels[train_indices],
        transform=transform_train
    )
    val_dataset = CIFAR10Dataset(
        train_data[val_indices],
        train_labels[val_indices],
        transform=transform_val
    )

    return train_dataset, val_dataset


def get_class_distribution(labels):
    """统计各类别样本数量"""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
