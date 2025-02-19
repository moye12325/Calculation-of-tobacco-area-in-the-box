import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
original = cv2.imread("image_0020.jpg")  # 替换为你的原始图像路径
segmentation = cv2.imread("image_0020_segmentation.png", cv2.IMREAD_GRAYSCALE)

# 调整大小
if original.shape[:2] != segmentation.shape[:2]:
    segmentation = cv2.resize(segmentation, (original.shape[1], original.shape[0]))

# 转换为彩色
segmentation_colored = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
segmentation_colored[np.where((segmentation_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # 白色变红色

# 叠加
alpha = 0.5  # 透明度
blended = cv2.addWeighted(original, 1 - alpha, segmentation_colored, alpha, 0)

# 用 matplotlib 显示
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Segmentation Overlay")
plt.show()