"""
全局配置文件
"""
import os
import torch

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "cifar-10-batches-py")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLIT_DIR = os.path.join(DATA_DIR, "split")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集配置
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 数据划分
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 训练配置
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# 学习率调度
LR_SCHEDULER = "StepLR"       # StepLR / ReduceLROnPlateau / CosineAnnealing
LR_STEP_SIZE = 15
LR_GAMMA = 0.5

# 超参数实验配置
LR_EXPERIMENT_VALUES = [0.01, 0.001, 0.0001]
BATCH_SIZE_EXPERIMENT_VALUES = [16, 32, 64]
DROPOUT_EXPERIMENT_VALUES = [0.2, 0.5, 0.7]

# DataLoader
NUM_WORKERS = 0  # Windows 建议设为 0

# 随机种子
SEED = 42
