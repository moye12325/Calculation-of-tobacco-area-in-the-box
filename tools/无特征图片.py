import numpy as np
import cv2

height, width = 2048, 3072  # 设定掩码尺寸
mask = np.zeros((height, width), dtype=np.uint8)  # 生成全黑掩码
cv2.imwrite("empty_mask.png", mask)  # 保存掩码