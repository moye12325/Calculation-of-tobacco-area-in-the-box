'''
File: model_loader.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 10:59
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''
import torch
from .NestedUNet import NestedUNet
from config.paths import MODEL_PATH
from config.parameters import NUM_CLASSES, INPUT_CHANNELS, DEVICE

def load_model():
    """加载 PyTorch 预训练模型"""
    model = NestedUNet(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    print(f"✅ Loaded model from: {MODEL_PATH}")
    return model