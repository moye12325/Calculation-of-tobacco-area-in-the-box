'''
File: my_dataset.py
Author: moye12325
Description: 1:图像分割数据集类
Created: $TIME
Version: v1.0

修改记录:
Date        Author        Modification Content
2025/2/19   moye12325     添加文件注释
'''

import os
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
from config import Config

# 新建联合数据增强函数（确保图像和 mask 同步增强）
def joint_transforms(image, mask, image_size):
    # 随机水平翻转
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    # 随机垂直翻转
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    # 随机旋转（例如：90 度倍数旋转）
    angle = random.choice([0, 90, 180, 270])
    if angle:
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    # 改变尺寸：图像用双线性插值，mask 用最近邻插值
    image = TF.resize(image, image_size, interpolation=transforms.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, image_size, interpolation=transforms.InterpolationMode.NEAREST)
    return image, mask


class ImageSegmentationDataset:
    def __init__(self, image_dir, mask_dir, file_list, transform_image=None, transform_mask=None, joint_transform=None, image_size=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.joint_transform = joint_transform  # 新增联合数据增强
        self.image_size = image_size  # 为了在 joint_transform 中使用

    def __getitem__(self, idx):
        image_file = self.file_list[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_file = image_file.replace(".jpg", "_mask.png")  # 确保 mask 文件命名一致
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道

        # 如果定义了联合变换，先对原始 PIL Image 进行操作
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask, Config.IMAGE_SIZE)

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        # mask 经 ToTensor 后 shape 为 [1, H, W]，squeeze 后变为 [H, W]
        return image, mask.squeeze(0)

    def __len__(self):
        return len(self.file_list)

# ======================= 修正 `transform_mask` 的示例（仅作为参考） =======================
# image_size = (256, 256)
# num_classes = 2  # 确保 `num_classes` 正确
#
# transform_image = transforms.Compose([
#     transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
#     transforms.ToTensor(),
# ])
#
# transform_mask = transforms.Compose([
#     transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
#     transforms.ToTensor(),
#     lambda x: (x * 255).long().clamp(0, num_classes - 1)  # ✅ 限制到 `[0, num_classes-1]`
# ])