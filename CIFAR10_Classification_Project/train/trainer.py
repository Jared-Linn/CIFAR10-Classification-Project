"""
训练器模块
"""
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from config.config import DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from train.early_stopping import EarlyStopping, ModelCheckpoint
from train.scheduler import create_scheduler


class Trainer:
    """
    通用训练器

    支持：
    - 训练/验证循环
    - Early Stopping
    - 学习率调度
    - 最佳模型保存
    - 指标记录
    """

    def __init__(self, model, train_loader, val_loader,
                 criterion=None, optimizer=None,
                 lr=0.001, weight_decay=1e-4,
                 scheduler_name="StepLR", scheduler_kwargs=None,
                 early_stopping_patience=5,
                 model_name="model"):
        """
        Args:
            model: PyTorch 模型
            train_loader: 训练集 DataLoader
            val_loader: 验证集 DataLoader
            criterion: 损失函数 (默认 CrossEntropyLoss)
            optimizer: 优化器 (默认 Adam)
            lr: 学习率
            weight_decay: 权重衰减
            scheduler_name: 学习率调度器名称
            scheduler_kwargs: 调度器参数字典
            early_stopping_patience: 早停耐心值
            model_name: 模型名称（用于保存 checkpoint）
        """
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name

        # 损失函数
        self.criterion = criterion or nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 学习率调度器
        sched_kwargs = scheduler_kwargs or {}
        self.scheduler, self.scheduler_uses_metric = create_scheduler(
            self.optimizer, scheduler_name, **sched_kwargs
        )

        # 早停 & 模型保存
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.checkpoint = ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth"), mode="max"
        )

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
        }
        self.best_val_acc = 0.0

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """验证一个 epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def train(self, num_epochs=50):
        """
        完整训练流程

        Args:
            num_epochs: 最大训练 epochs

        Returns:
            history: 训练历史字典
        """
        print(f"\n{'='*60}")
        print(f"Training: {self.model_name}")
        print(f"Device: {DEVICE}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Scheduler: {type(self.scheduler).__name__}")
        print(f"{'='*60}")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # 训练 & 验证
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler_uses_metric:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # 保存最佳模型
            self.checkpoint(val_acc, self.model)

            # 打印信息
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

            # 早停检查
            if self.early_stopping(val_loss, epoch):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        self.best_val_acc = max(self.history["val_acc"])
        print(f"\nTraining finished. Best Val Acc: {self.best_val_acc:.2f}%")
        return self.history
