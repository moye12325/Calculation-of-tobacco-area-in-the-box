'''
File: infer_and_save.py
Author: moye12325
Description: 1:模型训练
Created: $TIME
Version: v1.0

修改记录:
Date        Author        Modification Content
2025/2/19   moye12325     添加文件注释
'''


import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from my_dataset import ImageSegmentationDataset  # 自定义数据集
from NestedUNet import NestedUNet  # 模型定义文件
from sklearn.model_selection import train_test_split
from torchvision.transforms import InterpolationMode

# ======================= 1. 设备配置 =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 定义超参数
batch_size = 8
learning_rate = 1e-4
num_epochs = 200
num_classes = 2
patience = 10  # Early Stopping 的耐心值
weight_decay = 1e-4  # AdamW 正则化参数
image_size = (256, 256)  # 统一图像大小

# ======================= 3. 数据预处理 =======================
transform_image = transforms.Compose([
    transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    lambda x: (x * 255).long().clamp(0, num_classes - 1)  # 还原类别索引
])

# ======================= 4. 加载数据 =======================
image_dir = "./dataset/train/images"
mask_dir = "./dataset/train/masks"

# 获取所有图像文件
image_files = sorted(os.listdir(image_dir))

# 按 80% 训练，20% 验证划分
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

train_dataset = ImageSegmentationDataset(image_dir, mask_dir, train_files, transform_image, transform_mask)
val_dataset = ImageSegmentationDataset(image_dir, mask_dir, val_files, transform_image, transform_mask)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ======================= 5. 初始化模型 =======================
model = NestedUNet(num_classes=num_classes, input_channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# ======================= 6. 早停策略 =======================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=patience)


# ======================= 7. 训练函数 =======================
def train():
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 确保 `output` 形状是 `[batch_size, num_classes, H, W]`
            # print("Output Shape:", outputs.shape)

            # 处理深度监督
            if isinstance(outputs, list):
                loss = sum(criterion(out, masks) for out in outputs) / len(outputs)
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ======================= 8. 计算验证集损失 =======================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                if isinstance(outputs, list):
                    loss = sum(criterion(out, masks) for out in outputs) / len(outputs)
                else:
                    loss = criterion(outputs, masks)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # 学习率调度
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ======================= 9. 早停机制 =======================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./model_version_dir/best_model_V4_511.pth")  # 仅保存最佳模型
            print("Best model saved!")

        if early_stopping(avg_val_loss):
            print("Early stopping triggered!")
            break

# ======================= 10. 训练模型 =======================
if __name__ == "__main__":
    train()
