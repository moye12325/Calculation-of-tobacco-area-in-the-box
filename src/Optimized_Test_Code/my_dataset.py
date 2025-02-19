import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
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
        mask_file = image_file.replace(".jpg", "_mask.png")  # 确保 mask 文件命名一致
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask.squeeze(0)  # 确保 mask 形状为 [H, W]

    def __len__(self):
        # 返回数据集中图像文件的数量
        return len(self.file_list)

# ======================= 修正 `transform_mask` =======================
image_size = (256, 256)
num_classes = 2  # 确保 `num_classes` 正确

transform_image = transforms.Compose([
    transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    lambda x: (x * 255).long().clamp(0, num_classes - 1)  # ✅ 限制到 `[0, num_classes-1]`
])