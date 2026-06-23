"""
快速端到端测试 - 用少量 epoch 验证完整流水线
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互模式
import matplotlib.pyplot as plt
import torch

from config.config import RAW_DATA_DIR, OUTPUT_DIR, CLASS_NAMES, DEVICE
from utils.utils import set_seed
from utils.visualization import plot_class_distribution, plot_sample_images

from datasets.cifar10_dataset import load_cifar10_data, create_data_splits, CIFAR10Dataset, get_class_distribution
from datasets.transforms import get_train_transforms, get_val_transforms
from datasets.dataloader import create_dataloaders

from models.baseline_cnn import BaselineCNN, count_parameters
from models.resnet18_custom import ResNet18CIFAR

from train.trainer import Trainer
from evaluate.evaluate import evaluate_model
from evaluate.confusion_matrix import plot_confusion_matrix, find_worst_classes

set_seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("QUICK E2E TEST (2 epochs)")
print(f"Device: {DEVICE}")
print("=" * 60)

# Step 1: 数据加载和可视化
print("\n[1] Data loading...")
train_data, train_labels, test_data, test_labels = load_cifar10_data(RAW_DATA_DIR)
print(f"    Train: {train_data.shape}, Test: {test_data.shape}")

# 可视化
plot_class_distribution(np.concatenate([train_labels, test_labels]),
                        save_path=os.path.join(OUTPUT_DIR, "class_distribution.png"))
plot_sample_images(train_data[:500], train_labels[:500],
                   save_path=os.path.join(OUTPUT_DIR, "sample_images.png"))
print("    Charts saved: class_distribution.png, sample_images.png")

# 数据划分
train_dataset, val_dataset = create_data_splits(
    train_data, train_labels,
    transform_train=get_train_transforms(),
    transform_val=get_val_transforms()
)
test_dataset = CIFAR10Dataset(test_data, test_labels, transform=get_val_transforms())
print(f"    Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# Step 2: 快速训练
print("\n[2] Quick training (2 epochs)...")
_, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64)
train_loader, _, _ = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64)

# Baseline
print("\n--- Baseline CNN ---")
baseline = BaselineCNN().to(DEVICE)
baseline_trainer = Trainer(baseline, train_loader, val_loader,
                           lr=0.001, scheduler_name="StepLR",
                           scheduler_kwargs={"step_size": 10, "gamma": 0.5},
                           early_stopping_patience=5, model_name="test_baseline")
baseline_history = baseline_trainer.train(num_epochs=2)

# ResNet-18
print("\n--- ResNet-18 ---")
resnet = ResNet18CIFAR().to(DEVICE)
resnet_trainer = Trainer(resnet, train_loader, val_loader,
                          lr=0.001, scheduler_name="StepLR",
                          scheduler_kwargs={"step_size": 10, "gamma": 0.5},
                          early_stopping_patience=5, model_name="test_resnet")
resnet_history = resnet_trainer.train(num_epochs=2)

# Step 3: 评估
print("\n[3] Evaluation...")
baseline_metrics, b_y_true, b_y_pred = evaluate_model(baseline, test_loader)
resnet_metrics, r_y_true, r_y_pred = evaluate_model(resnet, test_loader)

# 混淆矩阵
plot_confusion_matrix(r_y_true, r_y_pred, CLASS_NAMES,
                      save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print("    Chart saved: confusion_matrix.png")

# 对比
print(f"\n{'='*50}")
print(f"{'Metric':<20} {'Baseline':<12} {'ResNet-18':<12}")
print(f"{'-'*50}")
for name in ["accuracy", "macro_f1", "weighted_f1"]:
    print(f"{name:<20} {baseline_metrics[name]:<12.4f} {resnet_metrics[name]:<12.4f}")

print(f"\n{'='*50}")
print("E2E TEST PASSED!")
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 50)

plt.close("all")
