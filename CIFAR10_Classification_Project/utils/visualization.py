"""
数据探索与可视化
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config.config import OUTPUT_DIR, CLASS_NAMES


def plot_class_distribution(labels, save_path=None):
    """
    绘制类别分布柱状图

    Args:
        labels: 标签数组
        save_path: 保存路径
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [CLASS_NAMES[i] for i in unique]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    bars = ax.bar(class_names, counts, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("CIFAR-10 Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    # 在柱子上标注数值
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(count), ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Class distribution saved to: {save_path}")
    return fig


def plot_sample_images(data, labels, save_path=None, num_per_class=5):
    """
    随机展示每个类别的样本图片

    Args:
        data: (N, 3, 32, 32) or (N, 3072) numpy array
        labels: (N,) numpy array
        save_path: 保存路径
        num_per_class: 每类展示数量
    """
    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(num_classes, num_per_class,
                              figsize=(num_per_class * 2, num_classes * 2.2))

    for cls_idx in range(num_classes):
        cls_samples = np.where(labels == cls_idx)[0]
        selected = np.random.choice(cls_samples, num_per_class, replace=False)

        for j, idx in enumerate(selected):
            img = data[idx]
            if img.ndim == 1:
                img = img.reshape(3, 32, 32)
            # 转置为 (H, W, C) for matplotlib
            img = img.transpose(1, 2, 0)
            # 反归一化（假设原始数据在 [0, 1]）
            img = np.clip(img, 0, 1)

            axes[cls_idx, j].imshow(img)
            axes[cls_idx, j].axis("off")
            if j == 0:
                axes[cls_idx, j].set_ylabel(CLASS_NAMES[cls_idx], fontsize=10,
                                             rotation=0, ha="right", va="center")

    fig.suptitle("CIFAR-10 Sample Images", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample images saved to: {save_path}")
    return fig


def plot_image_size_analysis(sizes, save_path=None):
    """
    分析图片尺寸分布（对于 CIFAR-10，所有图片都是 32x32）

    Args:
        sizes: 尺寸列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    heights, widths = zip(*sizes)

    axes[0].hist(heights, bins=20, color="#3498db", edgecolor="white", alpha=0.7)
    axes[0].axvline(np.mean(heights), color="red", linestyle="--", label=f"Mean: {np.mean(heights):.1f}")
    axes[0].set_xlabel("Height (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Image Height Distribution")
    axes[0].legend()

    axes[1].hist(widths, bins=20, color="#e74c3c", edgecolor="white", alpha=0.7)
    axes[1].axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.1f}")
    axes[1].set_xlabel("Width (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Image Width Distribution")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Image size analysis saved to: {save_path}")
    return fig
