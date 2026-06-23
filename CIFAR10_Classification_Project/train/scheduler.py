"""
学习率调度器
"""
import torch.optim as optim


def create_scheduler(optimizer, scheduler_name="StepLR", **kwargs):
    """
    创建学习率调度器

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: "StepLR", "ReduceLROnPlateau", "CosineAnnealing"
        **kwargs: 调度器参数

    Returns:
        scheduler, use_metric (bool): 调度器, 是否需要传入 metric
    """
    if scheduler_name == "StepLR":
        step_size = kwargs.get("step_size", 15)
        gamma = kwargs.get("gamma", 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, False

    elif scheduler_name == "ReduceLROnPlateau":
        patience = kwargs.get("patience", 5)
        factor = kwargs.get("factor", 0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, verbose=True
        )
        return scheduler, True  # 需要传入 val_loss

    elif scheduler_name == "CosineAnnealing":
        T_max = kwargs.get("T_max", 50)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return scheduler, False

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
