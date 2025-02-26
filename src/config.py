# File: config.py
"""
配置文件，所有路径、超参数、模型配置等公共参数统一在此设置
"""

class Config:
    # ======================= 路径配置 =======================
    DATA_PATHS = {
        "train_image_dir": "./dataset/train/images",
        "train_mask_dir": "./dataset/train/masks",
        "test_image_dirs": [
            './dataset/1-2000',
            './dataset/2001-4000',
            './dataset/4001-6000',
            './dataset/6001-8000',
            './dataset/8001-9663'
        ],
        "model_save_dir": "./model_version_dir",
        "segmentation_output_base": "./result/segmentation_results",
        "overlay_output_base": "./result/overlay_results"
    }

    # ======================= 模型配置 =======================
    MODEL_CONFIG = {
        # "architecture": "NestedUNet",
        "base_version": "V4",
        "num_classes": 2,
        "input_channels": 3
    }

    # ======================= 训练参数 =======================
    TRAIN_PARAMS = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 1000,
        "num_classes": 2,
        "patience": 10,
        "weight_decay": 1e-4,
        "image_size": (256, 256),
        "test_split": 0.2,
        "random_state": 42,
        "num_workers": 4,
        "model_version": "V5"
    }

    # ======================= 推理参数 =======================
    INFERENCE_PARAMS = {
        "inference_image_size": (256, 256),
        "num_workers": 8
    }

    IMAGE_SIZE=(256, 256)

# 单例配置对象
config = Config()
# print(config.INFERENCE_PARAMS.inference_image_size)