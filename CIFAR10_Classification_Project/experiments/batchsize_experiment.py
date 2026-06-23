"""
超参数实验 2：Batch Size 对比
对比 batch_size = {16, 32, 64} 对训练的影响
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from config.config import OUTPUT_DIR, BATCH_SIZE_EXPERIMENT_VALUES, DEVICE
from datasets.cifar10_dataset import load_cifar10_data, create_data_splits, CIFAR10Dataset
from datasets.transforms import get_train_transforms, get_val_transforms
from datasets.dataloader import create_dataloaders
from models.baseline_cnn import BaselineCNN
from train.trainer import Trainer


def run_batchsize_experiment():
    """运行 Batch Size 对比实验"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("Experiment 2: Batch Size Comparison")
    print("Testing batch_size =", BATCH_SIZE_EXPERIMENT_VALUES)
    print("=" * 60)

    # 加载数据
    train_data, train_labels, test_data, test_labels = load_cifar10_data()
    train_dataset, val_dataset = create_data_splits(
        train_data, train_labels,
        transform_train=get_train_transforms(),
        transform_val=get_val_transforms()
    )
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=get_val_transforms())

    results = {}
    training_times = {}
    best_accs = {}

    for bs in BATCH_SIZE_EXPERIMENT_VALUES:
        print(f"\n--- Training with batch_size={bs} ---")
        train_loader, val_loader, _ = create_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size=bs
        )

        model = BaselineCNN()
        trainer = Trainer(
            model, train_loader, val_loader,
            lr=0.001,
            scheduler_name="StepLR",
            scheduler_kwargs={"step_size": 15, "gamma": 0.5},
            early_stopping_patience=5,
            model_name=f"baseline_bs{bs}"
        )

        start_time = time.time()
        history = trainer.train(num_epochs=30)
        elapsed = time.time() - start_time

        results[bs] = history
        training_times[bs] = elapsed
        best_accs[bs] = max(history["val_acc"])

    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, bs in enumerate(BATCH_SIZE_EXPERIMENT_VALUES):
        h = results[bs]
        axes[0].plot(h["val_loss"], color=colors[i], label=f"BS={bs}", linewidth=2)
        axes[1].plot(h["val_acc"], color=colors[i], label=f"BS={bs}", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Batch Size - Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("Batch Size - Val Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 柱状图：Accuracy vs Training Time
    x = np.arange(len(BATCH_SIZE_EXPERIMENT_VALUES))
    width = 0.35
    acc_values = [best_accs[bs] for bs in BATCH_SIZE_EXPERIMENT_VALUES]
    time_values = [training_times[bs] for bs in BATCH_SIZE_EXPERIMENT_VALUES]
    labels = [f"BS={bs}" for bs in BATCH_SIZE_EXPERIMENT_VALUES]

    ax2 = axes[2]
    bars1 = ax2.bar(x - width / 2, acc_values, width, label="Best Acc (%)", color="#3498db")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width / 2, time_values, width, label="Time (s)", color="#e74c3c")
    ax2_twin.set_ylabel("Training Time (s)")
    ax2.set_title("Accuracy vs Training Time")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "batchsize_compare.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBatch size comparison chart saved to: {save_path}")

    # 打印汇总
    print("\n--- Batch Size Experiment Summary ---")
    for bs in BATCH_SIZE_EXPERIMENT_VALUES:
        print(f"BS={bs}: Best Acc = {best_accs[bs]:.2f}%, Time = {training_times[bs]:.1f}s")

    return results, training_times


if __name__ == "__main__":
    run_batchsize_experiment()
