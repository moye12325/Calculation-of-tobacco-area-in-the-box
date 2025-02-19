'''
File: parameters.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:02
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''

import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练参数
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
PATIENCE = 10
WEIGHT_DECAY = 1e-4

# 模型参数
NUM_CLASSES = 2
INPUT_CHANNELS = 3
IMAGE_SIZE = (256, 256)


