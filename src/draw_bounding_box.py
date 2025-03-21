'''
File: label_trans.py
Author: moye12325
Description: 1:画出边框计算面积
Created: $TIME
Version: v1.0

修改记录:
Date        Author        Modification Content
2025/2/19   moye12325     添加文件注释
'''


# 画边框算面积
import cv2
import numpy as np
import os
from config import Config
import glob

# 自动获取最新模型文件名
model_dir = Config.DATA_PATHS['model_save_dir']
model_files = glob.glob(os.path.join(model_dir, "*.pth"))

if not model_files:
    raise FileNotFoundError(f"No trained models found in {model_dir}")

# 按修改时间排序获取最新模型文件名
model_files.sort(key=os.path.getmtime, reverse=True)
model_filename = os.path.basename(model_files[0])

# 动态生成保存路径
segmentation_base_dir = f"./result/segmentation_results_{os.path.splitext(model_filename)[0]}"
calculated_base_dir = f"./result/calculated_results_{os.path.splitext(model_filename)[0]}"

# 创建基础输出目录
os.makedirs(calculated_base_dir, exist_ok=True)

# # 分割结果文件夹（输入）
# segmentation_dirs = [
#     './result/segmentation_results_2025-02-20_09-08/1-2000',
#     './result/segmentation_results_2025-02-20_09-08/2001-4000',
#     './result/segmentation_results_2025-02-20_09-08/4001-6000',
#     './result/segmentation_results_2025-02-20_09-08/6001-8000',
#     './result/segmentation_results_2025-02-20_09-08/8001-9663'
# ]
#
# # 计算结果保存的根目录
# output_root = "./result/calculated_results_2025-02-20_09-08"
# os.makedirs(output_root, exist_ok=True)  # 确保根目录存在

# 预定义的ROI边界框（可修改）
x, y, w, h = 68, 0, 180, 256
bounding_box_area = w * h  # 计算ROI区域的总像素数

# 动态获取所有分割文件夹
segmentation_dirs = [os.path.join(segmentation_base_dir, os.path.basename(dir_path)) for dir_path in Config.DATA_PATHS['test_image_dirs']]

# 遍历所有分割文件夹
for seg_dir in segmentation_dirs:
    # 计算对应的输出目录
    output_dir = os.path.join(calculated_base_dir, os.path.basename(seg_dir))
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    # 获取所有分割图片文件（假设是 PNG 格式）
    image_files = [f for f in os.listdir(seg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_filename in image_files:
        # 完整的输入路径
        image_path = os.path.join(seg_dir, image_filename)

        # 读取图像（灰度模式）
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像正确加载
        if image is None:
            print(f"Warning: Failed to load {image_path}, skipping...")
            continue  # 跳过当前文件

        # 裁剪边界框内的图像
        roi = image[y:y+h, x:x+w]

        # 统计白色像素的数量（白色像素值为255）
        white_pixel_count = np.sum(roi == 255)

        # 计算白色像素占比
        white_pixel_ratio = white_pixel_count / bounding_box_area

        # 转换为 BGR 以绘制彩色边界框
        image_with_box = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 绘制红色边界框
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 红色框 (BGR: 0,0,255)

        # 在图像上添加白色像素占比文本
        text = f"White: {white_pixel_ratio * 100:.2f}%"
        text_position = (x + 5, y + 20)  # 文本位置（稍微偏移以避免重叠边框）
        cv2.putText(image_with_box, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)  # 绿色文本 (BGR: 0,255,0)

        # 生成输出路径
        output_path = os.path.join(output_dir, f"calculated_{image_filename}")

        # 保存带有标注的图像
        cv2.imwrite(output_path, image_with_box)

        # 显示处理进度
        print(f"Processed {image_filename}: White Pixel Ratio = {white_pixel_ratio:.4f} ({white_pixel_ratio * 100:.2f}%)")
        print(f"Saved to: {output_path}")

print(f"\n✅ Batch processing completed! All calculated images saved in '{calculated_base_dir}' 🚀")