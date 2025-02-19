import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from my_dataset import ImageSegmentationDataset  # 自定义数据集
from NestedUNet import NestedUNet  # 模型定义文件
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as F

# ======================= 设备配置 =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
batch_size = 8
learning_rate = 1e-4
num_epochs = 200
num_classes = 2
patience = 10
weight_decay = 1e-4
crop_size = 512  # **修改为 512×512 训练**

# ======================= 数据预处理 =======================
def random_crop_with_overlap(image, mask, crop_size=512):
    """训练时随机裁剪 512×512 Patch"""
    h, w = image.shape[-2], image.shape[-1]
    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    image_crop = F.crop(image, y, x, crop_size, crop_size)
    mask_crop = F.crop(mask, y, x, crop_size, crop_size)
    return image_crop, mask_crop

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: random_crop_with_overlap(x, x, crop_size=crop_size))  # 训练时裁剪 512×512
])

transform_mask = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).long().clamp(0, num_classes - 1),
    transforms.Lambda(lambda x: random_crop_with_overlap(x, x, crop_size=crop_size))  # 训练时裁剪 512×512
])

# ======================= 数据加载 =======================
image_dir = "../dataset/train/images"
mask_dir = "../dataset/train/masks"
image_files = sorted(os.listdir(image_dir))
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

train_dataset = ImageSegmentationDataset(image_dir, mask_dir, train_files, transform_image, transform_mask)
val_dataset = ImageSegmentationDataset(image_dir, mask_dir, val_files, transform_image, transform_mask)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ======================= 初始化模型 =======================
model = NestedUNet(num_classes=num_classes, input_channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ======================= 训练函数 =======================
def train():
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}")

        # 保存最佳模型
        torch.save(model.state_dict(), "./opt_best_model.pth")

if __name__ == "__main__":
    train()