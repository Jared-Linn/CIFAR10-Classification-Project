"""
主入口脚本 - 完整实验流水线
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    OUTPUT_DIR, CHECKPOINT_DIR, RAW_DATA_DIR, CLASS_NAMES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE
)
from utils.utils import set_seed, format_time
from utils.visualization import plot_class_distribution, plot_sample_images, plot_image_size_analysis

from datasets.cifar10_dataset import (
    load_cifar10_data, create_data_splits, CIFAR10Dataset,
    get_class_distribution
)
from datasets.transforms import get_train_transforms, get_val_transforms
from datasets.dataloader import create_dataloaders

from models.baseline_cnn import BaselineCNN, count_parameters
from models.resnet18_custom import ResNet18CIFAR

from train.trainer import Trainer
from evaluate.evaluate import evaluate_model
from evaluate.confusion_matrix import plot_confusion_matrix, find_worst_classes


def step1_data_preparation():
    """步骤1：数据准备与可视化"""
    print("\n" + "=" * 70)
    print("STEP 1: Data Preparation & Exploration")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1.1 加载数据
    print("\n[1.1] Loading CIFAR-10 data...")
    train_data, train_labels, test_data, test_labels = load_cifar10_data(RAW_DATA_DIR)
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Train labels shape: {train_labels.shape}")
    print(f"  Test data shape: {test_data.shape}")
    print(f"  Test labels shape: {test_labels.shape}")

    # 1.2 数据探索与可视化
    print("\n[1.2] Data Exploration & Visualization...")

    # 类别分布
    dist = get_class_distribution(np.concatenate([train_labels, test_labels]))
    for cls_id, count in dist.items():
        print(f"  {CLASS_NAMES[cls_id]:>12s}: {count} samples")
    plot_class_distribution(
        np.concatenate([train_labels, test_labels]),
        save_path=os.path.join(OUTPUT_DIR, "class_distribution.png")
    )

    # 样本展示
    plot_sample_images(
        train_data[:500], train_labels[:500],
        save_path=os.path.join(OUTPUT_DIR, "sample_images.png"),
        num_per_class=5
    )

    # 尺寸分析 (CIFAR-10 都是 32x32)
    print("\n  Image size analysis:")
    sizes = [(32, 32)] * len(train_data)
    print(f"    Min size: (32, 32), Max size: (32, 32), Average size: (32.0, 32.0)")
    print(f"    All images are 32×32 pixels (standard CIFAR-10)")
    plot_image_size_analysis(
        sizes, save_path=os.path.join(OUTPUT_DIR, "image_size_analysis.png")
    )

    plt.close("all")

    # 1.3 数据划分
    print("\n[1.3] Creating data splits (Stratified Sampling)...")
    train_dataset, val_dataset = create_data_splits(
        train_data, train_labels,
        transform_train=get_train_transforms(),
        transform_val=get_val_transforms()
    )
    test_dataset = CIFAR10Dataset(
        test_data, test_labels, transform=get_val_transforms()
    )

    # 验证划分后各类别分布
    train_dist = get_class_distribution(train_dataset.labels)
    val_dist = get_class_distribution(val_dataset.labels)
    test_dist = get_class_distribution(test_dataset.labels)
    print(f"  Train set: {len(train_dataset)} samples")
    print(f"  Val set:   {len(val_dataset)} samples")
    print(f"  Test set:  {len(test_dataset)} samples")
    print(f"  Stratified sampling ensures consistent class ratios across splits.")

    return train_dataset, val_dataset, test_dataset


def step2_train_models(train_dataset, val_dataset):
    """步骤2：训练 Baseline CNN 和 ResNet-18"""
    print("\n" + "=" * 70)
    print("STEP 2: Model Training")
    print("=" * 70)

    # 创建 DataLoader (仅训练/验证)
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset, val_dataset, batch_size=BATCH_SIZE
    )

    print(f"\n  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")

    # 2.1 训练 Baseline CNN
    print("\n" + "-" * 40)
    print("[2.1] Training Baseline CNN")
    print("-" * 40)

    baseline_model = BaselineCNN()
    print(f"  Parameters: {count_parameters(baseline_model):,}")

    baseline_trainer = Trainer(
        baseline_model, train_loader, val_loader,
        lr=LEARNING_RATE,
        scheduler_name="StepLR",
        scheduler_kwargs={"step_size": 15, "gamma": 0.5},
        early_stopping_patience=5,
        model_name="baseline_cnn"
    )
    baseline_history = baseline_trainer.train(num_epochs=NUM_EPOCHS)

    # 2.2 训练 ResNet-18
    print("\n" + "-" * 40)
    print("[2.2] Training ResNet-18 (Improved Model)")
    print("-" * 40)

    resnet_model = ResNet18CIFAR()
    print(f"  Parameters: {count_parameters(resnet_model):,}")

    resnet_trainer = Trainer(
        resnet_model, train_loader, val_loader,
        lr=LEARNING_RATE,
        scheduler_name="StepLR",
        scheduler_kwargs={"step_size": 15, "gamma": 0.5},
        early_stopping_patience=5,
        model_name="resnet18"
    )
    resnet_history = resnet_trainer.train(num_epochs=NUM_EPOCHS)

    return baseline_trainer, resnet_trainer, train_loader, val_loader


def step3_evaluate(baseline_trainer, resnet_trainer, test_dataset):
    """步骤3：模型评估与对比"""
    print("\n" + "=" * 70)
    print("STEP 3: Model Evaluation & Comparison")
    print("=" * 70)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())

    # 3.1 Baseline 评估
    print("\n" + "-" * 40)
    print("[3.1] Baseline CNN Evaluation")
    print("-" * 40)
    baseline_model = baseline_trainer.model
    baseline_metrics, baseline_y_true, baseline_y_pred = evaluate_model(
        baseline_model, test_loader
    )

    # 绘制混淆矩阵
    baseline_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_baseline.png")
    plot_confusion_matrix(
        baseline_y_true, baseline_y_pred, CLASS_NAMES,
        save_path=baseline_cm_path,
        title="Baseline CNN - Confusion Matrix"
    )

    # 最差类别
    worst = find_worst_classes(baseline_metrics["per_class_f1"], CLASS_NAMES)
    print(f"\n  Worst classes (Baseline): {worst[0][0]} (F1={worst[0][1]:.4f}), {worst[1][0]} (F1={worst[1][1]:.4f})")

    # 3.2 ResNet-18 评估
    print("\n" + "-" * 40)
    print("[3.2] ResNet-18 Evaluation")
    print("-" * 40)
    resnet_model = resnet_trainer.model
    resnet_metrics, resnet_y_true, resnet_y_pred = evaluate_model(
        resnet_model, test_loader
    )

    resnet_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_resnet.png")
    plot_confusion_matrix(
        resnet_y_true, resnet_y_pred, CLASS_NAMES,
        save_path=resnet_cm_path,
        title="ResNet-18 - Confusion Matrix"
    )

    worst = find_worst_classes(resnet_metrics["per_class_f1"], CLASS_NAMES)
    print(f"\n  Worst classes (ResNet-18): {worst[0][0]} (F1={worst[0][1]:.4f}), {worst[1][0]} (F1={worst[1][1]:.4f})")

    # 3.3 双模型对比
    print("\n" + "-" * 40)
    print("[3.3] Model Comparison")
    print("-" * 40)

    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {'Baseline CNN':<18} {'ResNet-18':<18}")
    print(f"{'-'*60}")
    metrics_pairs = [
        ("Accuracy", baseline_metrics["accuracy"], resnet_metrics["accuracy"]),
        ("Macro-F1", baseline_metrics["macro_f1"], resnet_metrics["macro_f1"]),
        ("Weighted-F1", baseline_metrics["weighted_f1"], resnet_metrics["weighted_f1"]),
        ("Macro-Precision", baseline_metrics["macro_precision"], resnet_metrics["macro_precision"]),
        ("Macro-Recall", baseline_metrics["macro_recall"], resnet_metrics["macro_recall"]),
    ]
    for name, b_val, r_val in metrics_pairs:
        print(f"{name:<20} {b_val:<18.4f} {r_val:<18.4f}")

    print(f"\n{'='*60}")
    print(f"{'Resource':<20} {'Baseline CNN':<18} {'ResNet-18':<18}")
    print(f"{'-'*60}")
    print(f"{'Params':<20} {count_parameters(baseline_model):<18,} {count_parameters(resnet_model):<18,}")
    baseline_time = sum(baseline_trainer.history["epoch_time"])
    resnet_time = sum(resnet_trainer.history["epoch_time"])
    print(f"{'Training Time':<20} {format_time(baseline_time):<18} {format_time(resnet_time):<18}")

    # 3.4 绘制对比曲线
    plot_comparison_curves(baseline_trainer.history, resnet_trainer.history)

    plt.close("all")

    return baseline_metrics, resnet_metrics


def plot_comparison_curves(baseline_hist, resnet_hist):
    """绘制双模型对比曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs_b = range(1, len(baseline_hist["train_loss"]) + 1)
    epochs_r = range(1, len(resnet_hist["train_loss"]) + 1)

    # Loss 对比
    axes[0].plot(epochs_b, baseline_hist["train_loss"], "#e74c3c", linewidth=1.5, alpha=0.7, label="Baseline Train")
    axes[0].plot(epochs_b, baseline_hist["val_loss"], "#e74c3c", linewidth=2, linestyle="--", label="Baseline Val")
    axes[0].plot(epochs_r, resnet_hist["train_loss"], "#3498db", linewidth=1.5, alpha=0.7, label="ResNet Train")
    axes[0].plot(epochs_r, resnet_hist["val_loss"], "#3498db", linewidth=2, linestyle="--", label="ResNet Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves Comparison")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Accuracy 对比
    axes[1].plot(epochs_b, baseline_hist["train_acc"], "#e74c3c", linewidth=1.5, alpha=0.7, label="Baseline Train")
    axes[1].plot(epochs_b, baseline_hist["val_acc"], "#e74c3c", linewidth=2, linestyle="--", label="Baseline Val")
    axes[1].plot(epochs_r, resnet_hist["train_acc"], "#3498db", linewidth=1.5, alpha=0.7, label="ResNet Train")
    axes[1].plot(epochs_r, resnet_hist["val_acc"], "#3498db", linewidth=2, linestyle="--", label="ResNet Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves Comparison")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 验证 Loss 收敛对比
    axes[2].plot(epochs_b, baseline_hist["val_loss"], "#e74c3c", linewidth=2, label="Baseline CNN")
    axes[2].plot(epochs_r, resnet_hist["val_loss"], "#3498db", linewidth=2, label="ResNet-18")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation Loss")
    axes[2].set_title("Validation Loss Convergence")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存各个单独的图
    # Loss curve
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(epochs_b, baseline_hist["train_loss"], "#e74c3c", linewidth=1.5, alpha=0.7, label="Baseline Train")
    ax_loss.plot(epochs_b, baseline_hist["val_loss"], "#e74c3c", linewidth=2, linestyle="--", label="Baseline Val")
    ax_loss.plot(epochs_r, resnet_hist["train_loss"], "#3498db", linewidth=1.5, alpha=0.7, label="ResNet Train")
    ax_loss.plot(epochs_r, resnet_hist["val_loss"], "#3498db", linewidth=2, linestyle="--", label="ResNet Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig_loss.savefig(os.path.join(OUTPUT_DIR, "loss_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_loss)

    # Accuracy curve
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    ax_acc.plot(epochs_b, baseline_hist["train_acc"], "#e74c3c", linewidth=1.5, alpha=0.7, label="Baseline Train")
    ax_acc.plot(epochs_b, baseline_hist["val_acc"], "#e74c3c", linewidth=2, linestyle="--", label="Baseline Val")
    ax_acc.plot(epochs_r, resnet_hist["train_acc"], "#3498db", linewidth=1.5, alpha=0.7, label="ResNet Train")
    ax_acc.plot(epochs_r, resnet_hist["val_acc"], "#3498db", linewidth=2, linestyle="--", label="ResNet Val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Training & Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    fig_acc.savefig(os.path.join(OUTPUT_DIR, "accuracy_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_acc)

    plt.close("all")


def main():
    """主流水线"""
    set_seed(42)

    print("=" * 70)
    print("CIFAR-10 Image Classification: Model Design & Optimization")
    print("Neural Networks & Deep Learning - Final Project")
    print("=" * 70)

    start_time = time.time()

    # Step 1: 数据准备
    train_dataset, val_dataset, test_dataset = step1_data_preparation()

    # Step 2: 模型训练
    baseline_trainer, resnet_trainer, train_loader, val_loader = step2_train_models(
        train_dataset, val_dataset
    )

    # Step 3: 模型评估
    baseline_metrics, resnet_metrics = step3_evaluate(
        baseline_trainer, resnet_trainer, test_dataset
    )

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Project completed in {format_time(total_time)}")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
