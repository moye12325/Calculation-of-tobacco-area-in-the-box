'''
File: my_dataset.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:03
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''
import os
from torchvision import transforms
from PIL import Image

class ImageSegmentationDataset:
    def __init__(self, image_dir, mask_dir, file_list, transform_image=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __getitem__(self, idx):
        image_file = self.file_list[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_file = image_file.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask.squeeze(0)

    def __len__(self):
        return len(self.file_list)