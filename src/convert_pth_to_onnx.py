'''
File: infer_and_save.py
Author: moye12325
Description: 1:给推理后的图片加框，计算面积占比
Created: $TIME
Version: v1.0

修改记录:
Date        Author        Modification Content
2025/2/19   moye12325     添加文件注释
'''


import torch
import torch.nn as nn
import torch.onnx
from NestedUNet import NestedUNet  # 确保 NestedUNet.py 在同一目录下

# **1. 加载模型**
model_path = "./model_version_dir/best_model.pth"  # 你的 .pth 文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **2. 初始化 NestedUNet（确保参数与训练时一致）**
num_classes = 2  # 你训练时的类别数
input_channels = 3  # 你训练时的输入通道数
deep_supervision = False  # 你训练时是否使用深监督

model = NestedUNet(num_classes=num_classes, input_channels=input_channels, deep_supervision=deep_supervision)
model.to(device)

# **3. 加载权重**
checkpoint = torch.load(model_path, map_location=device)

# 检查是否是 state_dict 或完整模型
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])  # 适用于训练时保存了完整 checkpoint
else:
    model.load_state_dict(checkpoint)  # 适用于仅保存 state_dict

model.eval()  # 进入推理模式

# **4. 准备 ONNX 导出**
dummy_input = torch.randn(1, input_channels, 256, 256).to(device)  # 适配你的输入尺寸
onnx_path = "./model_version_dir/best_model.onnx"

# **5. 导出 ONNX**
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # 保存所有参数
    opset_version=11,  # 适用于 TensorRT / ONNX Runtime
    do_constant_folding=True,  # 进行常量折叠优化
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # 允许动态 batch
)

print(f"✅ 模型成功转换为 ONNX，保存至 {onnx_path}")