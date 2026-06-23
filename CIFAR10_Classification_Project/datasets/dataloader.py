"""
DataLoader 创建模块
"""
import torch
from torch.utils.data import DataLoader
from config.config import BATCH_SIZE, NUM_WORKERS


def create_dataloaders(train_dataset, val_dataset, test_dataset=None,
                       batch_size=BATCH_SIZE, num_workers=0):
    """
    创建训练、验证、测试 DataLoader

    Args:
        train_dataset: 训练集 Dataset
        val_dataset: 验证集 Dataset
        test_dataset: 测试集 Dataset (可选)
        batch_size: 批大小
        num_workers: 数据加载进程数 (Windows 建议设为 0)

    Returns:
        train_loader, val_loader, test_loader (test_loader 可能为 None)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader
