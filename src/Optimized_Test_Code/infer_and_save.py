import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from NestedUNet import NestedUNet  # 模型定义文件

# ======================= 设备配置 =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================= 加载模型 =======================
def load_model(model, path):
    """加载 PyTorch 预训练模型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from: {path}")
    return model

# ======================= 滑窗推理 =======================
def sliding_window_inference(image, model, tile_size=512, overlap=0.2):
    """滑窗推理，使用重叠区域加权平均融合"""
    h, w = image.shape[-2], image.shape[-1]
    stride = int(tile_size * (1 - overlap))

    output_mask = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[..., y:y_end, x:x_end].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tile)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()

            output_mask[y:y_end, x:x_end] += pred
            count_map[y:y_end, x:x_end] += 1

    output_mask /= count_map
    return output_mask

# ======================= 处理所有图片 =======================
def segment_images(model, image_dir, output_dir, tile_size=512, overlap=0.2):
    """滑窗推理并保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        filepath = os.path.join(image_dir, filename)
        image = Image.open(filepath).convert('RGB')
        image_tensor = transforms.ToTensor()(image).to(device)

        prediction = sliding_window_inference(image_tensor, model, tile_size=tile_size, overlap=overlap)

        output_filename = f"{os.path.splitext(filename)[0]}_segmentation.png"
        output_path = os.path.join(output_dir, output_filename)
        pred_img = (prediction * 255).astype(np.uint8)
        Image.fromarray(pred_img).save(output_path)

        print(f"✔ Processed: {filename} → {output_filename}")

if __name__ == "__main__":
    model = NestedUNet(num_classes=2, input_channels=3)
    model = load_model(model, "./best_model.pth")
    segment_images(model, "./dataset/test", "./dataset/segmentation_results")