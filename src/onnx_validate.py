'''
File: onnx_validate.py
Author: moye12325
Description: onnx模型验证
Created: $TIME
Version: v1.0

修改记录:
Date        Author        Modification Content
2025/2/19   moye12325     添加文件注释
'''


import onnxruntime as ort
import numpy as np

# **加载 ONNX**
ort_session = ort.InferenceSession("./model_version_dir/best_model.onnx")

# **生成随机输入**
input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
outputs = ort_session.run(None, {"input": input_data})

print("ONNX 推理结果:", outputs[0].shape)  # 形状应为 (1, num_classes, 256, 256)