import os
import cv2
import xml.etree.ElementTree as ET
import shutil

def create_voc_xml(image_path, annotations, save_path):
    root = ET.Element("annotation")

    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")

    img = cv2.imread(image_path)
    h, w, _ = img.shape
    width.text = str(w)
    height.text = str(h)

    for category, box in annotations:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = "char"  # 将类别名称更改为"char"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")

        xmin.text = str(int(box[0]))
        ymin.text = str(int(box[1]))
        xmax.text = str(int(box[2]))
        ymax.text = str(int(box[3]))

    tree = ET.ElementTree(root)
    tree.write(save_path)

def convert_to_voc(image_folder, annotation_folder, voc_image_folder, voc_annotation_folder):
    # 获取图像和注释文件列表
    image_files = os.listdir(image_folder)
    annotation_files = os.listdir(annotation_folder)

    # 创建 VOC 数据集的图像和注释文件夹
    os.makedirs(voc_image_folder, exist_ok=True)
    os.makedirs(voc_annotation_folder, exist_ok=True)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, image_file.replace(".jpg", ".txt"))

        # 读取图像
        img = cv2.imread(image_path)

        annotations = []
        with open(annotation_path, "r", encoding='utf-8') as f:  # 指定文件编码为'utf-8'
            for line in f:
                data = line.strip().split()
                category = data[0]
                box = [float(x) for x in data[1:]]
                annotations.append((category, box))

        # 创建 VOC 格式的 XML 文件
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        xml_path = os.path.join(voc_annotation_folder, image_name + ".xml")
        create_voc_xml(image_path, annotations, xml_path)

        # 复制图像到 VOC 数据集的图像文件夹中
        target_image_path = os.path.join(voc_image_folder, image_file)
        shutil.copy(image_path, target_image_path)


if __name__ == "__main__":
    # 定义输入文件夹和输出文件夹
    train_image_folder = "../data/MTH1000/train/train_images/"  # 训练集图像文件夹
    train_annotation_folder = "../data/MTH1000/train/train_annotations/"  # 训练集注释文件夹
    test_image_folder = "../data/MTH1000/test/test_images/"  # 测试集图像文件夹
    test_annotation_folder = "../data/MTH1000/test/test_annotations/"  # 测试集注释文件夹
    voc_train_image_folder = "voc_train_images"  # VOC 格式训练集图像文件夹
    voc_train_annotation_folder = "voc_train_annotations"  # VOC 格式训练集注释文件夹
    voc_test_image_folder = "voc_test_images"  # VOC 格式测试集图像文件夹
    voc_test_annotation_folder = "voc_test_annotations"  # VOC 格式测试集注释文件夹

    # 转换训练集到 VOC 格式
    convert_to_voc(train_image_folder, train_annotation_folder, voc_train_image_folder, voc_train_annotation_folder)

    # 转换测试集到 VOC 格式
    convert_to_voc(test_image_folder, test_annotation_folder, voc_test_image_folder, voc_test_annotation_folder)
