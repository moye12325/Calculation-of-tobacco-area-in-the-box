'''
File: segment_overlay.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:06
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''
import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from config.paths import TEST_IMAGE_DIR, SEGMENTATION_RESULTS_DIR, OVERLAY_RESULTS_DIR

os.makedirs(OVERLAY_RESULTS_DIR, exist_ok=True)

def process_overlay(image_file):
    """叠加原图和分割结果"""
    input_path = os.path.join(TEST_IMAGE_DIR, image_file)
    seg_path = os.path.join(SEGMENTATION_RESULTS_DIR, image_file.replace(".jpg", "_segmentation.png"))

    if not os.path.exists(seg_path):
        print(f"❌ Missing segmentation: {seg_path}")
        return

    original = cv2.imread(input_path)
    segmentation = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

    if original is None or segmentation is None:
        return

    segmentation_colored = cv2.merge([segmentation, segmentation, segmentation])
    segmentation_colored[segmentation == 255] = [0, 0, 255]

    blended = cv2.addWeighted(original, 0.5, segmentation_colored, 0.5, 0)
    output_path = os.path.join(OVERLAY_RESULTS_DIR, image_file)
    cv2.imwrite(output_path, blended)
    print(f"✔ Overlay saved: {output_path}")

def overlay_results():
    """批量处理叠加"""
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.png', '.jpg'))]

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_overlay, image_files)

if __name__ == "__main__":
    overlay_results()

