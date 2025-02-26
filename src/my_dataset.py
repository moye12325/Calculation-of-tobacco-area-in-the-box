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
import numpy as np
import cv2
import albumentations as A

# 新建联合数据增强函数（确保图像和 mask 同步增强）
# 新增参数 crop_size，如果传入则进行随机裁剪
def joint_transforms(image, mask, image_size, crop_size=None):
    """
    原始联合数据增强
    使用随机翻转、随机旋转、随机裁剪和统一 Resize 操作
    """
    # 随机水平翻转
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    # 随机垂直翻转
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    # 随机旋转（例如：0、90、180、270 度中的一个）
    angle = random.choice([0, 90, 180, 270])
    if angle:
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    # 若指定了随机裁剪尺寸，则进行随机裁剪
    if crop_size is not None:
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
    # 最后统一 Resize 到目标尺寸（图像用双线性插值，mask 用最近邻插值）
    image = TF.resize(image, image_size, interpolation=transforms.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, image_size, interpolation=transforms.InterpolationMode.NEAREST)
    return image, mask

def joint_transforms_albu(image, mask, image_size, crop_size=None):
    """
    新增联合数据增强函数
    使用 Albumentations 实现更多数据增强操作，同时对图像和 mask 保持同步变换

    参数:
      image: PIL Image (RGB)
      mask:  PIL Image (单通道)
      image_size: 目标尺寸 (height, width)
      crop_size: 随机裁剪尺寸，如果传入则优先执行随机裁剪
    """
    # 将 PIL Image 转换为 NumPy 数组
    image_np = np.array(image)
    mask_np = np.array(mask)

    # 构建增强流水线列表
    transform_list = [
        A.HorizontalFlip(p=0.5), # 以50%的概率对图像进行水平翻转（左右镜像）
        A.VerticalFlip(p=0.5), # 以50%的概率对图像进行垂直翻转（上下颠倒）。
        A.RandomRotate90(p=0.5), # 随机90度旋转

        # shift_limit=0.0625: 平移范围为图像宽/高的±6.25%。
        # scale_limit=0.1: 随机缩放图像，范围为原始大小的 ±10%（即 0.9~1.1 倍）。
        # rotate_limit=45: 随机旋转角度范围为±45度。
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),

        # scale=(0.05, 0.1): 控制透视形变强度，值越大形变越明显。
        # 作用: 以50%概率对图像进行3D视角变换（如远小近大效果）。
        A.Perspective(scale=(0.05, 0.1), p=0.5),

        # alpha=1: 形变强度，值越大扭曲越明显。
        # sigma=50: 控制形变的平滑度，值越大形变越平缓。
        # 作用: 以50%概率生成类似“局部拉伸”或“褶皱”的弹性形变
        A.ElasticTransform(alpha=1, sigma=50, p=0.5)
    ]

    # 如果指定裁剪尺寸，执行随机裁剪
    # 随机裁剪到指定的尺寸，裁剪操作一定会执行（p=1.0）。
    if crop_size is not None:
        transform_list.append(A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0))

    # 使用 A.OneOf 从以下两种颜色调整操作中随机选择一种
    # 1. A.ColorJitter: 随机调整亮度±20%、对比度±20%、饱和度±20%和色调±0.1
    # 2. A.RandomBrightnessContrast: 随机调整亮度±20%和对比度±20%
    transform_list.append(A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
    ], p=0.5))

    # 添加噪声和模糊操作
    # 以 30% 的概率对图像进行高斯模糊，模糊核大小随机在 3×3 到 7×7 之间
    # 以 30% 的概率向图像添加高斯噪声，噪声方差范围为 10 到 50。
    transform_list.extend([
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
    ])

    # 最后统一 Resize 到目标尺寸（image 使用双线性插值）
    transform_list.append(A.Resize(height=image_size[0], width=image_size[1],
                                   interpolation=cv2.INTER_LINEAR, always_apply=True))

    # 组合所有转换，禁用尺寸检查（若你确定 image 与 mask 尺寸在前面已对齐）
    transform = A.Compose(transform_list, additional_targets={'mask': 'mask'}, is_check_shapes=False)
    augmented = transform(image=image_np, mask=mask_np)

    # 转换回 PIL Image 后返回
    image_aug = Image.fromarray(augmented['image'])
    mask_aug = Image.fromarray(augmented['mask'])
    return image_aug, mask_aug

class ImageSegmentationDataset:
    def __init__(self, image_dir, mask_dir, file_list, transform_image=None, transform_mask=None,
                 joint_transform=None, image_size=None, crop_size=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.joint_transform = joint_transform  # 联合数据增强函数
        self.image_size = image_size           # 目标尺寸
        self.crop_size = crop_size             # 随机裁剪尺寸

    def __getitem__(self, idx):
        image_file = self.file_list[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_file = image_file.replace(".jpg", "_mask.png")  # 确保 mask 文件命名一致
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道 mask

        # 根据是否设置联合数据增强选择处理方式
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask, self.image_size, crop_size=self.crop_size)

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        # mask 经 ToTensor 后 shape 为 [1, H, W]，squeeze 后变为 [H, W]
        return image, mask.squeeze(0)

    def __len__(self):
        return len(self.file_list)
