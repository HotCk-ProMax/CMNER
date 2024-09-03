import os
import shutil

def check(folder_path):
    # 定义需要检查的文件名模板
    required_files = [
        "crop_location_{index}.jpg",
        "crop_miscellaneous_{index}.jpg",
        "crop_organization_{index}.jpg",
        "crop_person_{index}.jpg"
    ]

    # 获取所有子文件夹的路径
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 检查每个子文件夹
    for subfolder in subfolders:
        # 获取当前子文件夹的index
        index = os.path.basename(subfolder)

        missing_files = []
        existing_files = {}

        # 根据当前index检查每个所需文件是否存在
        for file_template in required_files:
            file_name = file_template.format(index=index)
            file_path = os.path.join(subfolder, file_name)

            if not os.path.exists(file_path):
                missing_files.append(file_name)
            else:
                existing_files[file_name] = file_path

        # 如果存在缺失文件
        if missing_files:
            # 找到一个已存在的文件进行复制
            if existing_files:
                source_file = list(existing_files.values())[0]  # 使用第一个已存在的文件
                for missing_file in missing_files:
                    destination_file = os.path.join(subfolder, missing_file)
                    shutil.copy(source_file, destination_file)
                    print(f"Copy {source_file} to {destination_file} to supplement missing files")
            else:
                print(f"Nothing in {subfolder}, please check by yourself")

    print("All checked!")
