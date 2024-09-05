import os
from PIL import Image


def convert_images_to_jpg(folder_path):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否为图像文件
        if os.path.isfile(file_path):
            try:
                # 打开图像文件
                with Image.open(file_path) as img:
                    # 获取文件名（不带扩展名）
                    base_name = os.path.splitext(filename)[0]
                    # 将图像转换为 RGB 模式（确保可以保存为 JPG）
                    img = img.convert("RGB")
                    # 保存为 JPG 格式
                    img.save(os.path.join(folder_path, base_name + ".jpg"), "JPEG")

                    # 删除原始文件（如果需要）
                    os.remove(file_path)
            except Exception as e:
                print(f"无法转换文件 {filename}：{e}")


# 指定要转换的文件夹路径
folder_path = "../data/MTH1000/train/train_images"  # 或 "./imgs/test"

# 执行转换
convert_images_to_jpg(folder_path)

print("图像转换完成。")
