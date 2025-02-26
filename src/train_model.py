'''
File: train_model.py
Author: moye12325
Description: 1:模型训练
Created: $TIME
Version: v1.0

修改记录:
Date        Author        Modification Content
2025/2/19   moye12325     添加文件注释
'''
from datetime import datetime
import re
import os
from config import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from my_dataset import ImageSegmentationDataset, joint_transforms, joint_transforms_albu  # 新增 joint_transforms_albu
from NestedUNet import NestedUNet  # 模型定义文件
from sklearn.model_selection import train_test_split
from functools import partial

# ======================= 1. 设备配置 =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 超参数
params = {
    "batch_size": Config.TRAIN_PARAMS['batch_size'],
    "learning_rate": Config.TRAIN_PARAMS['learning_rate'],
    "num_epochs": Config.TRAIN_PARAMS['num_epochs'],
    "num_classes": Config.TRAIN_PARAMS['num_classes'],
    "patience": Config.TRAIN_PARAMS['patience'],
    "weight_decay": Config.TRAIN_PARAMS['weight_decay'],
    "image_size": Config.IMAGE_SIZE,
    "model_version": Config.TRAIN_PARAMS['model_version'],  # 🔴 手动更改大版本号（v1 → v2）
    # 新增随机裁剪尺寸（目标尺寸的 80%）
    "crop_size": (int(Config.IMAGE_SIZE[0] * 0.8), int(Config.IMAGE_SIZE[1] * 0.8)),
    # 新增开关：是否使用新版的丰富联合数据增强
    "use_advanced_joint_augmentation": True
}

# 根据开关选择联合数据增强方式
if params.get("use_advanced_joint_augmentation", False):
    joint_transform_fn = joint_transforms_albu
else:
    joint_transform_fn = joint_transforms

# ======================= 3. 数据预处理 =======================
# -------------- 训练集预处理（先使用联合数据增强，再进行 ToTensor 与归一化） --------------
train_transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 使用 partial 将 crop_size 参数传入 joint_transforms，仅用于训练集
# train_joint_transform = partial(joint_transforms, crop_size=params["crop_size"])

# 训练集 mask 仅进行 ToTensor 和标签转换（joint_transforms 已经 Resize 过）
train_transform_mask = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255).long().clamp(0, params["num_classes"] - 1)
])

# -------------- 验证集预处理（直接使用 Resize, ToTensor 与归一化） --------------
val_transform_image = transforms.Compose([
    transforms.Resize(params["image_size"], interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transform_mask = transforms.Compose([
    transforms.Resize(params["image_size"], interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    lambda x: (x * 255).long().clamp(0, params["num_classes"] - 1)
])

# ======================= 4. 加载数据 =======================
image_dir = Config.DATA_PATHS['train_image_dir']
mask_dir = Config.DATA_PATHS['train_mask_dir']

# 获取所有图像文件
image_files = sorted(os.listdir(image_dir))

# 按 80% 训练，20% 验证划分
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# train_dataset = ImageSegmentationDataset(image_dir, mask_dir, train_files, transform_image, transform_mask)
# val_dataset = ImageSegmentationDataset(image_dir, mask_dir, val_files, transform_image, transform_mask)


# 创建数据集：训练集启用联合数据增强（由 joint_transform_fn 控制），验证集仅使用预处理转换
train_dataset = ImageSegmentationDataset(
    image_dir, mask_dir, train_files,
    transform_image=train_transform_image,
    transform_mask=train_transform_mask,
    joint_transform=joint_transform_fn,
    image_size=params["image_size"],
    # crop_size=params["crop_size"]
)
val_dataset = ImageSegmentationDataset(
    image_dir, mask_dir, val_files,
    transform_image=val_transform_image,
    transform_mask=val_transform_mask
    # 验证集不使用联合增强，直接由 transform 完成 Resize
)

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ======================= 5. 初始化模型 =======================
model = NestedUNet(num_classes=params["num_classes"], input_channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
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

early_stopping = EarlyStopping(patience=params["patience"])


# ======================= 版本管理函数 =======================
def get_next_model_version(model_dir, base_version):
    """
    自动增加小版本号，例如：
    - 当前目录下 `v1.0` 存在，则生成 `v1.1`
    - `v1.1` 存在，则生成 `v1.2`
    """
    existing_versions = []

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for filename in os.listdir(model_dir):
        match = re.search(rf"{base_version}\.(\d+)", filename)
        if match:
            existing_versions.append(int(match.group(1)))

    if existing_versions:
        new_version = f"{base_version}.{max(existing_versions) + 1}"
    else:
        new_version = f"{base_version}.0"

    return new_version

def get_loss_optimizer_abbr(loss_fn, optimizer):
    """获取损失函数和优化器的缩写"""
    loss_abbr = {
        "CrossEntropyLoss": "CE",
        "MSELoss": "MSE",
        "DiceLoss": "Dice",
        "BCELoss": "BCE"
    }.get(loss_fn.__class__.__name__, "Loss")

    optim_abbr = {
        "SGD": "SGD",
        "Adam": "Adam",
        "AdamW": "AdamW",
        "RMSprop": "RMS"
    }.get(optimizer.__class__.__name__, "Opt")

    return loss_abbr, optim_abbr

#======================= 生成时间戳（精确到小时和分钟）
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

# 获取训练集和验证集的图片数量
num_train_images = len(train_dataset)
num_val_images = len(val_dataset)

# 计算新的版本号
model_dir = Config.DATA_PATHS['model_save_dir']
# model_dir = "./model_version_dir"
new_version = get_next_model_version(model_dir, params["model_version"])

# 生成模型文件名

# 生成输入尺寸字符串（修正 {} 问题）
input_size_str = f"{params['image_size'][0]}x{params['image_size'][1]}"
# 获取损失函数和优化器缩写
loss_abbr, optim_abbr = get_loss_optimizer_abbr(criterion, optimizer)
# 生成模型文件名
model_filename = f"NestedUNet_{num_train_images}-{num_val_images}_{input_size_str}_{loss_abbr}_{optim_abbr}_{timestamp}_{new_version}.pth"
model_path = os.path.join(model_dir, model_filename)

# ======================= 7. 训练函数 =======================
def train():
    best_val_loss = float("inf")

    for epoch in range(params["num_epochs"]):
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

        print(f"Epoch [{epoch + 1}/{params['num_epochs']}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ======================= 9. 早停机制 =======================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)  # 仅保存最佳模型
            print("✅ Best model saved! " + model_filename)

        if early_stopping(avg_val_loss):
            print("✔✔✔✔✔ Early stopping triggered!")
            break


# ======================= 10. 训练模型 =======================
if __name__ == "__main__":
    train()