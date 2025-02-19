'''
File: data_loader.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:03
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''

import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from data_loader.my_dataset import ImageSegmentationDataset
from config.paths import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR, VAL_MASK_DIR
from config.parameters import BATCH_SIZE, IMAGE_SIZE, NUM_CLASSES

# 预处理
transform_image = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    lambda x: (x * 255).long().clamp(0, NUM_CLASSES - 1)
])

def get_dataloaders():
    train_files = sorted(os.listdir(TRAIN_IMAGE_DIR))
    val_files = sorted(os.listdir(VAL_IMAGE_DIR))

    train_dataset = ImageSegmentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, train_files, transform_image, transform_mask)
    val_dataset = ImageSegmentationDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, val_files, transform_image, transform_mask)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader