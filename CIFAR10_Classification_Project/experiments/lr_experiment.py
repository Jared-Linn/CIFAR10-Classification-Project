"""
超参数实验 1：学习率对比
对比 lr = {1e-2, 1e-3, 1e-4} 对模型收敛的影响
"""
import os
import matplotlib.pyplot as plt
from config.config import OUTPUT_DIR, LR_EXPERIMENT_VALUES, DEVICE
from datasets.cifar10_dataset import load_cifar10_data, create_data_splits, CIFAR10Dataset
from datasets.transforms import get_train_transforms, get_val_transforms
from datasets.dataloader import create_dataloaders
from models.baseline_cnn import BaselineCNN
from train.trainer import Trainer


def run_lr_experiment():
    """运行学习率对比实验"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("Experiment 1: Learning Rate Comparison")
    print("Testing lr =", LR_EXPERIMENT_VALUES)
    print("=" * 60)

    # 加载数据
    train_data, train_labels, test_data, test_labels = load_cifar10_data()
    train_dataset, val_dataset = create_data_splits(
        train_data, train_labels,
        transform_train=get_train_transforms(),
        transform_val=get_val_transforms()
    )
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=get_val_transforms())
    train_loader, val_loader, _ = create_dataloaders(train_dataset, val_dataset, test_dataset)

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for lr in LR_EXPERIMENT_VALUES:
        print(f"\n--- Training with lr={lr} ---")
        model = BaselineCNN()
        trainer = Trainer(
            model, train_loader, val_loader,
            lr=lr,
            scheduler_name="StepLR",
            scheduler_kwargs={"step_size": 15, "gamma": 0.5},
            early_stopping_patience=5,
            model_name=f"baseline_lr{lr}"
        )
        history = trainer.train(num_epochs=30)
        results[lr] = history

    # 绘制对比图
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, lr in enumerate(LR_EXPERIMENT_VALUES):
        h = results[lr]
        axes[0].plot(h["val_loss"], color=colors[i], label=f"lr={lr}", linewidth=2)
        axes[1].plot(h["val_acc"], color=colors[i], label=f"lr={lr}", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Learning Rate Comparison - Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("Learning Rate Comparison - Val Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "lr_compare.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nLearning rate comparison chart saved to: {save_path}")

    # 打印汇总
    print("\n--- Learning Rate Experiment Summary ---")
    for lr in LR_EXPERIMENT_VALUES:
        best_acc = max(results[lr]["val_acc"])
        print(f"lr={lr:.4f}: Best Val Acc = {best_acc:.2f}%")

    return results


if __name__ == "__main__":
    run_lr_experiment()
