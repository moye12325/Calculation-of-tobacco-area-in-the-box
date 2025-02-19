'''
File: paths
Author: 19423
Description: 优化代码，降低耦合
Created: 10:51
Version: v1.0

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''

import os
from glob import glob

# 根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集路径
DATASET_DIR = os.path.join(ROOT_DIR, "src", "dataset")
IMAGE_DIRS = [
    os.path.join(DATASET_DIR, "1-2000"),
    os.path.join(DATASET_DIR, "2001-4000"),
    os.path.join(DATASET_DIR, "4001-6000"),
    os.path.join(DATASET_DIR, "6001-8000"),
    os.path.join(DATASET_DIR, "8001-9663"),
]

# 模型路径
MODEL_DIR = os.path.join(ROOT_DIR, "src", "model_version_dir")
MODEL_PATH = sorted(glob(os.path.join(MODEL_DIR, "best_model_*.pth")))[-1]  # 自动获取最新模型
ONNX_PATH = os.path.join(MODEL_DIR, "best_model.onnx")

# 结果路径
RESULT_DIR = os.path.join(ROOT_DIR, "src", "result")
SEGMENTATION_RESULTS_DIR = os.path.join(RESULT_DIR, "segmentation_results")
OVERLAY_RESULTS_DIR = os.path.join(RESULT_DIR, "overlay_results")
CALCULATED_RESULTS_DIR = os.path.join(RESULT_DIR, "calculated_results")

# 确保目录存在
for directory in [MODEL_DIR, RESULT_DIR, SEGMENTATION_RESULTS_DIR, OVERLAY_RESULTS_DIR, CALCULATED_RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)