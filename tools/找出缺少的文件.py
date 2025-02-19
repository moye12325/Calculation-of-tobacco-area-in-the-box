import os

# 定义文件夹路径
images_dir = "/home/amax/PyProjects/src/dataset/train/images/"
masks_dir = "/home/amax/PyProjects/src/dataset/train/masks/"

# 获取 images 目录中的所有图片（去掉扩展名）
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))}

# 获取 masks 目录中的所有掩码文件（去掉 "_mask.png" 后的文件名）
mask_files = {f.replace("_mask", "").rsplit('.', 1)[0] for f in os.listdir(masks_dir) if f.endswith('_mask.png')}

# 找出 masks 目录中缺少的文件
missing_masks = image_files - mask_files

# 输出缺失的文件名
if missing_masks:
    print("缺少的 mask 文件对应的图片：")
    for missing in sorted(missing_masks):
        print(missing)
else:
    print("所有 mask 文件都完整！")