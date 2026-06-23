"""
早停机制 (Early Stopping)
"""
import torch


class EarlyStopping:
    """
    监控验证集 loss，若连续 patience 个 epoch 无改善则停止训练

    Args:
        patience: 容忍 epochs 数
        min_delta: 最小改善阈值
        verbose: 是否打印信息
    """

    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop


class ModelCheckpoint:
    """保存最佳模型权重"""

    def __init__(self, filepath, mode="min"):
        self.filepath = filepath
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")

    def __call__(self, metric, model):
        improved = False
        if self.mode == "min":
            improved = metric < self.best_metric
        else:
            improved = metric > self.best_metric

        if improved:
            self.best_metric = metric
            torch.save(model.state_dict(), self.filepath)
        return improved
