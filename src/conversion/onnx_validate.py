'''
File: onnx_validate.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:06
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''

import onnxruntime as ort
import numpy as np
from config.paths import ONNX_PATH

def validate_onnx():
    """验证 ONNX 推理"""
    ort_session = ort.InferenceSession(ONNX_PATH)
    input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
    outputs = ort_session.run(None, {"input": input_data})

    print(f"✅ ONNX 推理成功，输出形状: {outputs[0].shape}")

if __name__ == "__main__":
    validate_onnx()
