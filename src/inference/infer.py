'''
File: infer.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:05
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from model.model_loader import load_model
from config.paths import TEST_IMAGE_DIR, SEGMENTATION_RESULTS_DIR
from config.parameters import DEVICE, IMAGE_SIZE

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

def process_image(model, image_path, output_dir):
    """推理单张图片并保存分割结果"""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy() * 255

    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_segmentation.png"))
    Image.fromarray(prediction.astype(np.uint8)).save(output_path)
    print(f"✔ Processed: {os.path.basename(image_path)} → {output_path}")

def segment_images():
    """批量推理图像"""
    model = load_model()
    os.makedirs(SEGMENTATION_RESULTS_DIR, exist_ok=True)

    image_files = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.png', '.jpg'))]

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda img: process_image(model, img, SEGMENTATION_RESULTS_DIR), image_files)

if __name__ == "__main__":
    segment_images()
    print(f"✅ Segmentation completed! Results saved in {SEGMENTATION_RESULTS_DIR}")

