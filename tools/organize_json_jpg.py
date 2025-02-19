import os
import shutil

def organize_files(source_folder, json_dest_folder, jpg_dest_folder):
    """
    将存在成对的 .jpg 和 .json 文件移动到新的目标文件夹中。

    :param source_folder: 原始文件夹路径
    :param json_dest_folder: 保存 .json 文件的目标文件夹路径
    :param jpg_dest_folder: 保存 .jpg 文件的目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(json_dest_folder, exist_ok=True)
    os.makedirs(jpg_dest_folder, exist_ok=True)

    # 获取文件夹中所有文件
    files = os.listdir(source_folder)

    # 遍历文件夹，寻找成对文件
    for file in files:
        # 如果是 .json 文件
        if file.endswith('.json'):
            # 获取文件的基础名称（去掉扩展名）
            base_name = os.path.splitext(file)[0]
            # 对应的 .jpg 文件是否存在
            jpg_file = f"{base_name}.jpg"
            if jpg_file in files:
                # 移动 .json 文件
                json_src_path = os.path.join(source_folder, file)
                json_dest_path = os.path.join(json_dest_folder, file)
                shutil.move(json_src_path, json_dest_path)

                # 移动 .jpg 文件
                jpg_src_path = os.path.join(source_folder, jpg_file)
                jpg_dest_path = os.path.join(jpg_dest_folder, jpg_file)
                shutil.move(jpg_src_path, jpg_dest_path)

                print(f"Moved: {file} and {jpg_file} to respective folders.")

if __name__ == "__main__":
    # 指定源文件夹和目标文件夹
    source_folder = "./6001-8000"      # 替换为你的源文件夹路径
    json_dest_folder = "json_folder"    # 替换为保存 .json 文件的目标文件夹路径
    jpg_dest_folder = "jpg_folder"      # 替换为保存 .jpg 文件的目标文件夹路径

    organize_files(source_folder, json_dest_folder, jpg_dest_folder)