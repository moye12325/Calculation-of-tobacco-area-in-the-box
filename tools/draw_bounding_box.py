import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像（以灰度模式加载）
image_path = "image_0020_segmentation.png"  # 你的分割结果图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 确保图像正确加载
if image is None:
    raise FileNotFoundError(f"Error: File '{image_path}' not found or failed to load.")

# 设定已知的边界框坐标
x, y, w, h = 68, 0, 180, 256

# 计算边界框的面积
bounding_box_area = w * h

# 裁剪边界框内的图像
roi = image[y:y+h, x:x+w]

# 统计白色像素的数量（像素值为255）
white_pixel_count = np.sum(roi == 255)

# 计算白色像素占比
white_pixel_ratio = white_pixel_count / bounding_box_area

# 转换为 BGR 以绘制彩色边界框
image_with_box = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 绘制红色边界框
cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 颜色 (BGR: 0,0,255)

# 在图像上添加白色像素占比文本
text = f"White: {white_pixel_ratio * 100:.2f}%"
text_position = (x + 5, y + 20)  # 文本位置（稍微偏移以避免重叠边框）
cv2.putText(image_with_box, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2, cv2.LINE_AA)  # 绿色文本 (BGR: 0,255,0)

# 显示结果
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))  # OpenCV BGR 转 RGB 以正确显示
plt.title(f"White Pixel Ratio: {white_pixel_ratio:.4f}")
plt.axis("off")
plt.show()

# 保存带有标注的图像
output_path = "segmentation_with_box_ratio.png"
cv2.imwrite(output_path, image_with_box)

# 输出结果
print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")
print(f"Bounding Box Area: {bounding_box_area} pixels")
print(f"White Pixel Count: {white_pixel_count} pixels")
print(f"White Pixel Ratio: {white_pixel_ratio:.4f} (or {white_pixel_ratio * 100:.2f}%)")
print(f"Processed image saved at: {output_path}")