"""
模型评估指标模块
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)


def compute_metrics(y_true, y_pred, class_names=None):
    """
    计算分类评估指标

    Args:
        y_true: 真实标签 (N,)
        y_pred: 预测标签 (N,)
        class_names: 类别名称列表

    Returns:
        metrics: 指标字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # 每个类别的 F1 分数
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    if class_names:
        metrics["per_class_f1"] = dict(zip(class_names, per_class_f1))
    else:
        metrics["per_class_f1"] = {str(i): v for i, v in enumerate(per_class_f1)}

    return metrics


def get_classification_report(y_true, y_pred, class_names):
    """生成详细的分类报告"""
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
