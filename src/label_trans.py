import json
import numpy as np
import cv2
import os
from PIL import Image

def json_to_mask(json_file):
    """将 LabelMe 的 JSON 文件转换为掩码图像"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 读取图像的尺寸
    height, width = data['imageHeight'], data['imageWidth']

    # 创建一个空的掩码图像（初始化为全零，表示背景）
    mask = np.zeros((height, width), dtype=np.uint8)

    # 遍历 JSON 文件中的所有标注
    for shape in data['shapes']:
        points = shape['points']

        # 确保标注的点数至少为 3
        if len(points) >= 3:
            # 将点转换为整数
            polygon = np.array(points, dtype=np.int32)

            # 填充多边形到掩码
            cv2.fillPoly(mask, [polygon], color=1)  # 用 1 填充前景区域

    return mask

def save_mask(image_file, json_file, output_dir):
    """将图像文件的掩码保存到输出目录"""
    mask = json_to_mask(json_file)  # 生成掩码

    # 获取文件名（去掉扩展名）
    base_name, ext = os.path.splitext(os.path.basename(image_file))

    # 防止重复添加 `_mask`
    if not base_name.endswith('_mask'):
        mask_filename = f"{base_name}_mask.png"
    else:
        mask_filename = f"{base_name}.png"  # 防止重复

    mask_output_path = os.path.join(output_dir, mask_filename)

    # 将掩码保存为 PNG 图像（将 1 转换为 255，使其成为白色）
    cv2.imwrite(mask_output_path, mask * 255)
    print(f"保存掩码图像：{mask_output_path}")

def convert_json_to_masks(image_dir, json_dir, output_dir):
    """遍历图像目录，将所有 JSON 标注文件转换为掩码图像并保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.png')):
            json_file = os.path.join(json_dir, os.path.splitext(image_file)[0] + '.json')

            if os.path.exists(json_file):
                save_mask(image_file, json_file, output_dir)
            else:
                print(f"未找到对应的 JSON 文件：{json_file}")

# 设置路径
image_dir = 'dataset/train/images'  # 图像文件夹
json_dir = 'dataset/train/jsons'  # LabelMe 输出的 JSON 文件夹
output_dir = 'dataset/train/masks'  # 保存掩码的文件夹

# 转换所有 JSON 文件为掩码
convert_json_to_masks(image_dir, json_dir, output_dir)