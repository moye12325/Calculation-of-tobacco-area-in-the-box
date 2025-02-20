'''
File: utils.py
Author: 19423
Description: 
Created: 14:25
Version: v1.0

修改记录:
Date        Author        Modification Content
14:25    19423       Create the file
'''

# File: utils.py
"""
工具函数模块，包含版本管理、路径处理等公共函数
"""
import os
import re
from datetime import datetime
from config import config

def extract_model_timestamp(model_path):
    """
    从模型路径提取训练时间戳
    示例输入：NestedUNet_..._2025-02-20_10-53_V4.4.pth
    示例输出：2025-02-20_10-53
    """
    pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}"
    match = re.search(pattern, model_path)
    if not match:
        raise ValueError(f"模型路径中未找到有效时间戳: {model_path}")
    return match.group()

def generate_timestamp():
    """生成统一时间戳（与训练代码一致）"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def get_next_model_version():
    """
    自动生成模型版本号（示例：V4.0 → V4.1）
    """
    existing_versions = []
    pattern = re.compile(rf"{config.MODEL_CONFIG['base_version']}\.(\d+)")

    for filename in os.listdir(config.DATA_PATHS["model_save_dir"]):
        match = pattern.search(filename)
        if match:
            existing_versions.append(int(match.group(1)))

    if existing_versions:
        new_version = f"{config.MODEL_CONFIG['base_version']}.{max(existing_versions)+1}"
    else:
        new_version = f"{config.MODEL_CONFIG['base_version']}.0"

    return new_version

def generate_output_dir(base_path):
    """
    生成带时间戳的输出目录
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return os.path.join(base_path, f"results_{timestamp}")

def get_latest_model_path():
    """
    自动获取最新模型路径
    """
    pattern = re.compile(rf'_{re.escape(config.MODEL_CONFIG["base_version"])}\.(\d+)\.pth$')
    max_version = -1
    latest_model = None

    for filename in os.listdir(config.DATA_PATHS["model_save_dir"]):
        match = pattern.search(filename)
        if match:
            version = int(match.group(1))
            if version > max_version:
                max_version = version
                latest_model = filename

    if latest_model:
        return os.path.join(config.DATA_PATHS["model_save_dir"], latest_model)
    raise FileNotFoundError("No valid model found")