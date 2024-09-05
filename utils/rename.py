import os
import shutil


def rename_and_organize_files(root_dir):
    # 定义路径
    annotations_dir = os.path.join(root_dir, 'annotations')
    imgs_dir = os.path.join(root_dir, 'imgs')
    train_img_dir = os.path.join(imgs_dir, 'train')
    test_img_dir = os.path.join(imgs_dir, 'test')

    # 创建新annotations文件夹结构
    train_annotations_dir = os.path.join(annotations_dir, 'train')
    test_annotations_dir = os.path.join(annotations_dir, 'test')
    os.makedirs(train_annotations_dir, exist_ok=True)
    os.makedirs(test_annotations_dir, exist_ok=True)

    # 处理train文件夹
    rename_and_move_files(train_img_dir, train_annotations_dir, annotations_dir)

    # 处理test文件夹
    rename_and_move_files(test_img_dir, test_annotations_dir, annotations_dir)


def rename_and_move_files(img_dir, new_annotations_dir, original_annotations_dir):
    # 获取图像文件列表
    img_files = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

    # 重命名和移动文件
    for idx, img_file in enumerate(img_files):
        # 新的文件名
        new_img_name = f'img_{idx + 1:02d}.png'
        original_img_path = os.path.join(img_dir, img_file)
        new_img_path = os.path.join(img_dir, new_img_name)

        # 获取对应的标注文件
        base_name = os.path.splitext(img_file)[0]
        annotation_file = f'{base_name}.txt'
        original_annotation_path = os.path.join(original_annotations_dir, annotation_file)
        new_annotation_name = f'img_{idx + 1:02d}.txt'
        new_annotation_path = os.path.join(new_annotations_dir, new_annotation_name)

        # 重命名图像文件
        os.rename(original_img_path, new_img_path)

        # 移动对应的标注文件
        if os.path.exists(original_annotation_path):
            shutil.move(original_annotation_path, new_annotation_path)


# 指定项目根目录
project_root = '../data/MTH1000'  # 将此路径替换为你的项目根目录
rename_and_organize_files(project_root)
