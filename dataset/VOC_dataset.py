import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import random


def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = ('char')

    def __init__(self, root_dir, resize_size=[800, 1333], use_difficult=False, is_train=True, augment=None,
                 img_ids=None):
        self.root = root_dir
        self.use_difficult = use_difficult

        img_folder = os.path.join(root_dir, "img")
        if img_ids is None:
            self.img_ids = [filename.split('.')[0] for filename in os.listdir(img_folder) if filename.endswith('.jpg')]
        else:
            self.img_ids = img_ids

        self.name2id = dict(zip(VOCDataset.CLASSES_NAME, range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.resize_size = resize_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = Image.open(os.path.join(self.root, "img", "%s.jpg" % img_id))

        anno = ET.parse(os.path.join(self.root, "xml", "%s.xml" % img_id)).getroot()
        boxes = []
        classes = []
        for obj in anno.iter("object"):
            _box = obj.find("bndbox")
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE = 1
            box = tuple(map(lambda x: x - TO_REMOVE, list(map(float, box))))
            boxes.append(box)

            name = obj.find("name").text.lower().strip()
            classes.append(self.name2id.get(name, 0))

            # Check if "difficult" element exists before accessing its text
            difficult_element = obj.find("difficult")
            difficult = int(difficult_element.text) if difficult_element is not None else 0

        boxes = np.array(boxes, dtype=np.float32)
        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.augment is not None:
                img, boxes = self.augment(img, boxes)
        img = np.array(img)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        if len(image.shape) == 2:
            # 如果图像是灰度图像，则将通道数手动设置为 3
            image = np.stack([image] * 3, axis=-1)

        h, w,_ = image.shape
        min_side, max_side = input_ksize
        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        image_padded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_padded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_padded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_padded, boxes

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes


# # 初始化训练数据集
# train_dataset = VOCDataset(root_dir='../data/MTH1200/train/', resize_size=[1600, 1600],
#                            use_difficult=False, is_train=True, augment=None)

# # 打印训练数据集大小
# print("Train dataset size:", len(train_dataset))
#
# # 可视化一个训练样本
# sample_index = 0
# sample_img, sample_boxes, sample_classes = train_dataset[sample_index]
#
# # 绘制图像和边界框
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8, 8))
# plt.imshow(sample_img.permute(1, 2, 0))
# for box in sample_boxes:
#     xmin, ymin, xmax, ymax = box.tolist()
#     plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='r')
# plt.title("Sample Image with Bounding Boxes (Training)")
# plt.axis('off')
# plt.show()
