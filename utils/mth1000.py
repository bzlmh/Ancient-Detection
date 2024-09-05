import os
import shutil

# 原始文件夹路径
annotations_folder = "../data/MTH1000/annotations"
imgs_folder = "../data/MTH1000/imgs"
train_images_src = os.path.join(imgs_folder, "train")
test_images_src = os.path.join(imgs_folder, "test")

# 目标文件夹路径
train_image_folder = "../data/MTH1000/train/train_images"
train_annotation_folder = "../data/MTH1000/train/train_annotations"
test_image_folder = "../data/MTH1000/test/test_images"
test_annotation_folder = "../data/MTH1000/test/test_annotations"

# 创建目标文件夹
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_annotation_folder, exist_ok=True)
os.makedirs(test_image_folder, exist_ok=True)
os.makedirs(test_annotation_folder, exist_ok=True)


def move_files(image_src_folder, image_dst_folder, annotation_dst_folder):
    # 获取该图像文件夹下的所有图像文件
    image_files = [f for f in os.listdir(image_src_folder) if os.path.isfile(os.path.join(image_src_folder, f))]

    for image_file in image_files:
        image_name, _ = os.path.splitext(image_file)
        annotation_file = f"{image_name}.txt"

        # 确保图像文件和对应的标注文件存在
        annotation_path = os.path.join(annotations_folder, annotation_file)
        if os.path.exists(annotation_path):
            # 移动图像文件
            shutil.move(os.path.join(image_src_folder, image_file), os.path.join(image_dst_folder, image_file))
            # 移动标注文件
            shutil.move(annotation_path, os.path.join(annotation_dst_folder, annotation_file))


# 移动训练集文件
move_files(train_images_src, train_image_folder, train_annotation_folder)

# 移动测试集文件
move_files(test_images_src, test_image_folder, test_annotation_folder)

print("文件已成功移动。")
