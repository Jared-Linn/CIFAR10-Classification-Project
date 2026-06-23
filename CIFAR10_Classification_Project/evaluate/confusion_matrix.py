"""
混淆矩阵模块
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None,
                          normalize=True, title="Confusion Matrix"):
    """
    绘制并保存混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
        normalize: 是否归一化
        title: 图标题
    """
    cm = sk_confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float32") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def find_worst_classes(per_class_f1, class_names, top_k=2):
    """
    找出 F1 分数最低的类别

    Returns:
        list of (class_name, f1_score)
    """
    sorted_cls = sorted(per_class_f1.items(), key=lambda x: x[1])
    return sorted_cls[:top_k]
