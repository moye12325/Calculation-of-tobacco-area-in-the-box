import cv2
import os
import numpy as np
import concurrent.futures  # 线程池

# 原始图片文件夹
input_dirs = [
    # './dataset/1-2000',
    # './dataset/2001-4000',
    # './dataset/4001-6000',
    # './dataset/6001-8000',
    # './dataset/8001-9663',
    './dataset/train/images'
    ]

# 分割结果文件夹
segmentation_dirs = [
    # './result/segmentation_results/1-2000',
    # './result/segmentation_results/2001-4000',
    # './result/segmentation_results/4001-6000',
    # './result/segmentation_results/6001-8000',
    # './result/segmentation_results/8001-9663'
    './dataset/train/masks'
]

# 结果保存的文件夹
output_root = './data_verify/overlay_datas'
os.makedirs(output_root, exist_ok=True)

# 处理单张图片的函数
def process_image(input_dir, seg_dir, output_dir, image_file):
    input_path = os.path.join(input_dir, image_file)

    # 获取文件名（不带扩展名），如 "image_8880"
    file_name = os.path.splitext(image_file)[0]

    # 构造分割图的文件名，例如 "image_8880_segmentation.png"
    seg_file = f"{file_name}_mask.png"
    seg_path = os.path.join(seg_dir, seg_file)

    # 确保分割图片存在
    if not os.path.exists(seg_path):
        print(f"Warning: Mask file {seg_path} not found, skipping...")
        return

    # 读取原始图像 & 分割图像
    original = cv2.imread(input_path)
    segmentation = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

    # 确保图片成功加载
    if original is None or segmentation is None:
        print(f"Error: Failed to load {input_path} or {seg_path}, skipping...")
        return

    # 调整分割图大小以匹配原图
    if original.shape[:2] != segmentation.shape[:2]:
        segmentation = cv2.resize(segmentation, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 转换为彩色图像，并高效转换白色区域为红色
    segmentation_colored = cv2.merge([segmentation, segmentation, segmentation])  # 直接复制通道
    mask = segmentation == 255  # 生成白色区域的 mask
    segmentation_colored[mask] = [0, 0, 255]  # 直接赋值替换

    # 叠加分割结果（透明度 0.5）
    blended = cv2.addWeighted(original, 0.5, segmentation_colored, 0.5, 0)

    # 保存结果
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, blended)

    print(f"Processed: {image_file} -> {output_path}")

# 遍历所有文件夹，并行处理
for input_dir, seg_dir in zip(input_dirs, segmentation_dirs):
    output_dir = os.path.join(output_root, os.path.basename(input_dir))
    os.makedirs(output_dir, exist_ok=True)  # 创建对应的输出目录

    # 获取当前目录下的所有原始图片文件
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 使用多线程处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_image, input_dir, seg_dir, output_dir, image_file) for image_file in image_files]
        concurrent.futures.wait(futures)  # 等待所有任务完成

print("\n✅ Batch processing completed! All overlay images saved in './overlay_results/' 🚀")