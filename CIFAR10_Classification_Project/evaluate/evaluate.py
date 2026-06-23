"""
模型评估主模块
"""
import torch
import numpy as np
from config.config import DEVICE, CLASS_NAMES, NUM_CLASSES
from evaluate.metrics import compute_metrics, get_classification_report
from evaluate.confusion_matrix import plot_confusion_matrix


@torch.no_grad()
def evaluate_model(model, test_loader, class_names=CLASS_NAMES):
    """
    在测试集上全面评估模型

    Args:
        model: 训练好的 PyTorch 模型
        test_loader: 测试集 DataLoader
        class_names: 类别名称

    Returns:
        metrics: 评估指标字典
        y_true_all, y_pred_all: 真实/预测标签 (用于后续绘图)
    """
    model.eval()
    model.to(DEVICE)

    all_preds = []
    all_labels = []

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 计算指标
    metrics = compute_metrics(y_true, y_pred, class_names)

    # 打印结果
    print(f"\n{'='*50}")
    print(f"Model Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"\nPer-Class F1:")
    for cls_name, f1 in metrics["per_class_f1"].items():
        print(f"  {cls_name:>12s}: {f1:.4f}")

    print(f"\nClassification Report:")
    print(get_classification_report(y_true, y_pred, class_names))

    return metrics, y_true, y_pred
