"""
超参数实验 3：Dropout Rate 对比
对比 dropout = {0.2, 0.5, 0.7} 对过拟合的抑制效果
"""
import os
import matplotlib.pyplot as plt
from config.config import OUTPUT_DIR, DROPOUT_EXPERIMENT_VALUES, DEVICE
from datasets.cifar10_dataset import load_cifar10_data, create_data_splits, CIFAR10Dataset
from datasets.transforms import get_train_transforms, get_val_transforms
from datasets.dataloader import create_dataloaders
from models.resnet18_custom import ResNet18CIFAR
from train.trainer import Trainer


def run_dropout_experiment():
    """运行 Dropout Rate 对比实验"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("Experiment 3: Dropout Rate Comparison")
    print("Testing dropout =", DROPOUT_EXPERIMENT_VALUES)
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

    for dropout in DROPOUT_EXPERIMENT_VALUES:
        print(f"\n--- Training with dropout={dropout} ---")
        model = ResNet18CIFAR(dropout_rate=dropout)
        trainer = Trainer(
            model, train_loader, val_loader,
            lr=0.001,
            scheduler_name="StepLR",
            scheduler_kwargs={"step_size": 15, "gamma": 0.5},
            early_stopping_patience=5,
            model_name=f"resnet_dp{dropout}"
        )
        history = trainer.train(num_epochs=30)
        results[dropout] = history

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c", "#f39c12", "#3498db"]
    for i, dp in enumerate(DROPOUT_EXPERIMENT_VALUES):
        h = results[dp]
        axes[0].plot(h["val_loss"], color=colors[i], label=f"Dropout={dp}", linewidth=2)
        axes[1].plot(h["val_acc"], color=colors[i], label=f"Dropout={dp}", linewidth=2)

        # 同时标注 train/val gap
        train_final = h["train_acc"][-1]
        val_final = h["val_acc"][-1]
        gap = train_final - val_final
        print(f"Dropout={dp}: Train Acc={train_final:.2f}%, Val Acc={val_final:.2f}%, Gap={gap:.2f}%")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Dropout Rate Comparison - Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("Dropout Rate Comparison - Val Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "dropout_compare.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nDropout comparison chart saved to: {save_path}")

    # 打印汇总
    print("\n--- Dropout Experiment Summary ---")
    for dp in DROPOUT_EXPERIMENT_VALUES:
        best_acc = max(results[dp]["val_acc"])
        print(f"dropout={dp}: Best Val Acc = {best_acc:.2f}%")

    return results


if __name__ == "__main__":
    run_dropout_experiment()
