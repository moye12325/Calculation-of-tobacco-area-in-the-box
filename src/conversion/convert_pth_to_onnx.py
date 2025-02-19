'''
File: convert_pth_to_onnx.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:06
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''

import torch
import torch.onnx
from model.model_loader import load_model
from config.paths import ONNX_PATH, DEVICE, INPUT_CHANNELS

def export_onnx():
    """导出模型为 ONNX 格式"""
    model = load_model()
    dummy_input = torch.randn(1, INPUT_CHANNELS, 256, 256).to(DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"✅ 模型成功转换为 ONNX，保存至 {ONNX_PATH}")

if __name__ == "__main__":
    export_onnx()

