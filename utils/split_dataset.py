import os
import random
import shutil

def split_dataset(image_folder, annotation_folder, train_ratio):
    # 获取所有图像和注释文件的路径
    image_files = os.listdir(image_folder)
    annotation_files = os.listdir(annotation_folder)

    # 确定训练集和测试集的样本数量
    total_samples = len(image_files)
    train_samples = int(total_samples * train_ratio)
    test_samples = total_samples - train_samples

    # 随机选择训练集和测试集的样本
    train_images = random.sample(image_files, train_samples)

    # 将剩余的图像分配给测试集
    test_images = [img for img in image_files if img not in train_images]

    # 创建训练集和测试集的文件夹
    os.makedirs("../data/degard118/train_images", exist_ok=True)
    os.makedirs("../data/degard118/train_annotations", exist_ok=True)
    os.makedirs("../data/degard118/test_images", exist_ok=True)
    os.makedirs("../data/degard118/test_annotations", exist_ok=True)

    # 复制图像和注释文件到对应的文件夹中
    for img in train_images:
        shutil.copy(os.path.join(image_folder, img), "../data/degard118/train_images")
        shutil.copy(os.path.join(annotation_folder, img.replace(".jpg", ".txt")), "../data/MTH1200/train_annotations")

    for img in test_images:
        shutil.copy(os.path.join(image_folder, img), "../data/MTH1200/test_images")
        shutil.copy(os.path.join(annotation_folder, img.replace(".jpg", ".txt")), "../data/MTH1200/test_annotations")

if __name__ == "__main__":
    image_folder = "../data/MTH1200/img"  # 存储所有图像的文件夹路径
    annotation_folder = "../data/MTH1200/label_char"  # 存储所有注释文件的文件夹路径
    train_ratio = 0.8  # 训练集所占比例

    # 分割数据集
    split_dataset(image_folder, annotation_folder, train_ratio)
