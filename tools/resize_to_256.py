import os
from PIL import Image

# 输入和输出文件夹路径
input_folder = "input_images"   # 原始图片所在文件夹
output_folder = "output_images" # 处理后图片的保存文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 目标尺寸
target_size = (256, 256)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 确保是图片文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
        try:
            # 打开图片
            img = Image.open(input_path)

            # 调整大小（保持原图比例缩放并填充）
            img = img.resize(target_size, Image.ANTIALIAS)

            # 保存图片
            img.save(output_path)
            print(f"✅ {filename} 处理完成，已保存至 {output_folder}")

        except Exception as e:
            print(f"❌ 处理 {filename} 时发生错误: {e}")

print("🎉 所有图片处理完成！")