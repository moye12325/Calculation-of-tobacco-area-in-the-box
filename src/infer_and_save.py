import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from NestedUNet import NestedUNet
import warnings
from deprecated.sphinx import deprecated

## 推理代码

# ======================= 设备配置 =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================= 加载模型 =======================
def load_model(model, path):
    """加载 PyTorch 预训练模型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # 如果 best_model.pth 里包含额外的权重（如 Deep Supervision 层），可能会加载失败。即使 state_dict 里有额外的层，也不会报错。
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from: {path}")
    return model

# ======================= 进行推理 =======================
def segment_images(model, image_dir, output_dir, image_size=(256, 256)):
    """
    读取 `image_dir` 中的所有图片，进行分割，并保存到 `output_dir`
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义图像预处理
    transform = transforms.Compose([
        # torchvision.transforms.Resize() 默认插值 方式可能不稳定
        # ✅ 双线性插值
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

    # 处理所有图片
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"⚠️ No valid images found in {image_dir}")
        return

    print(f"Processing {len(image_files)} images in {image_dir}...")

    for filename in image_files:
        filepath = os.path.join(image_dir, filename)

        try:
            # 读取图像 + 维度扩展
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                # ✅ 进行推理 GPU 上计算 `argmax`
                outputs = model(input_tensor)
                prediction = torch.argmax(outputs, dim=1).squeeze(0)

            # 保存分割结果
            output_filename = f"{os.path.splitext(filename)[0]}_segmentation.png"
            output_path = os.path.join(output_dir, output_filename)

            # ✅ 直接转换 `uint8`
            pred_img = prediction.byte().cpu().numpy() * 255
            Image.fromarray(pred_img).save(output_path)

            print(f"✔ Processed: {filename} → {output_filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# 进行推理
@deprecated
def segment_images_old(model, image_dir, output_dir):
    # 计算新尺寸，将原始尺寸除以2
    new_height = 256
    new_width = 256

    # 数据预处理和数据增强
    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor()
    ])

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(image_dir, filename)
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)  # 增加批量维度

            with torch.no_grad():
                outputs = model(input_tensor)
                prediction = torch.argmax(outputs, dim=1).squeeze(0)  # 获取分割结果

            # 保存分割结果
            output_filename = filename.split('.')[0] + '_segmentation.png'
            output_path = os.path.join(output_dir, output_filename)

            # 将类别值映射到 0-255 范围
            pred_img = prediction.cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(pred_img).save(output_path)

# 主执行代码
if __name__ == "__main__":

    model_path = './model_version_dir/best_model_V4_511.pth'

    # ✅ 确保 `num_classes` 和 `input_channels` 设置正确
    model = NestedUNet(num_classes=2, input_channels=3)
    model = load_model(model, model_path)

    # 定义输入目录和输出目录
    input_dirs = [
        './dataset/1-2000',
        './dataset/2001-4000',
        './dataset/4001-6000',
        './dataset/6001-8000',
        './dataset/8001-9663'
    ]

    base_output_dir = './result/segmentation_results_V4_511'  # 基础输出结果目录

    for input_dir in input_dirs:
        output_dir = os.path.join(base_output_dir, os.path.basename(input_dir))
        segment_images(model, input_dir, output_dir)

    print(f"Segmentation results saved to: {base_output_dir}")
