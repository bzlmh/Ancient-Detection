# import random
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from torchvision.datasets import CocoDetection
# from torchvision import transforms
# import cv2
# from PIL import Image
#
#
# def flip(img, boxes):
#     img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     w = img.width
#     if boxes.shape[0] != 0:
#         xmin = w - boxes[:, 2]
#         xmax = w - boxes[:, 0]
#         boxes[:, 2] = xmax
#         boxes[:, 0] = xmin
#     return img, boxes
#
#
# class COCODataset(CocoDetection):
#     def __init__(self, imgs_path, anno_path, resize_size=[800, 800], is_train=True, transform=None):
#         super().__init__(imgs_path, anno_path)
#         print("INFO====>check annos, filtering invalid data......")
#         ids = []
#         for id in self.ids:
#             ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
#             ann = self.coco.loadAnns(ann_id)
#             if self._has_valid_annotation(ann):
#                 ids.append(id)
#         self.ids = ids
#         self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
#         self.id2category = {v: k for k, v in self.category2id.items()}
#
#         self.transform = transform
#         self.resize_size = resize_size
#
#         self.mean = [0.40789654, 0.44719302, 0.47026115]
#         self.std = [0.28863828, 0.27408164, 0.27809835]
#         self.train = is_train
#
#     def __getitem__(self, index):
#         img, ann = super().__getitem__(index)
#
#         ann = [o for o in ann if o['iscrowd'] == 0]
#         boxes = [o['bbox'] for o in ann]
#         boxes = np.array(boxes, dtype=np.float32)
#         boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]  # xywh-->xyxy
#
#         if self.train:
#             if random.random() < 0.5:
#                 img, boxes = flip(img, boxes)
#             if self.transform is not None:
#                 img, boxes = self.transform(img, boxes)
#
#         img = np.array(img)
#         img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)
#
#         classes = [o['category_id'] for o in ann]
#         classes = [self.category2id[c] for c in classes]
#
#         img = transforms.ToTensor()(img)
#         boxes = torch.from_numpy(boxes)
#         classes = torch.LongTensor(classes)
#
#         return img, boxes, classes
#
#     def preprocess_img_boxes(self, image, boxes, input_ksize):
#         min_side, max_side = input_ksize
#         h, w, _ = image.shape
#
#         smallest_side = min(w, h)
#         largest_side = max(w, h)
#         scale = min_side / smallest_side
#         if largest_side * scale > max_side:
#             scale = max_side / largest_side
#         nw, nh = int(scale * w), int(scale * h)
#         image_resized = cv2.resize(image, (nw, nh))
#
#         pad_w = 32 - nw % 32
#         pad_h = 32 - nh % 32
#
#         image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
#         image_paded[:nh, :nw, :] = image_resized
#
#         if boxes is None:
#             return image_paded
#         else:
#             boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
#             boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
#             return image_paded, boxes
#
#     def _has_only_empty_bbox(self, annot):
#         return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)
#
#     def _has_valid_annotation(self, annot):
#         if len(annot) == 0:
#             return False
#         if self._has_only_empty_bbox(annot):
#             return False
#         return True
#
#     def collate_fn(self, data):
#         imgs_list, boxes_list, classes_list = zip(*data)
#         assert len(imgs_list) == len(boxes_list) == len(classes_list)
#         batch_size = len(boxes_list)
#         pad_imgs_list = []
#         pad_boxes_list = []
#         pad_classes_list = []
#
#         h_list = [int(s.shape[1]) for s in imgs_list]
#         w_list = [int(s.shape[2]) for s in imgs_list]
#         max_h = np.array(h_list).max()
#         max_w = np.array(w_list).max()
#         for i in range(batch_size):
#             img = imgs_list[i]
#             pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
#                 torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.0)
#             ))
#
#         max_num = 0
#         for i in range(batch_size):
#             n = boxes_list[i].shape[0]
#             if n > max_num: max_num = n
#         for i in range(batch_size):
#             pad_boxes_list.append(
#                 torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
#             pad_classes_list.append(
#                 torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))
#
#         batch_boxes = torch.stack(pad_boxes_list)
#         batch_classes = torch.stack(pad_classes_list)
#         batch_imgs = torch.stack(pad_imgs_list)
#
#         return batch_imgs, batch_boxes, batch_classes
#
#
# def visualize(dataset, index=None):
#     if index is None:
#         index = random.randint(0, len(dataset) - 1)
#
#     img, boxes, classes = dataset[index]
#
#     # Convert tensor to numpy array and transpose to (H, W, C)
#     img = img.permute(1, 2, 0).numpy()
#
#     if isinstance(boxes, torch.Tensor):
#         boxes = boxes.numpy()
#     if isinstance(classes, torch.Tensor):
#         classes = classes.numpy()
#
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)
#
#     for box in boxes:
#         if box[0] < 0:
#             continue
#         rect = patches.Rectangle(
#             (box[0], box[1]), box[2] - box[0], box[3] - box[1],
#             linewidth=2, edgecolor='r', facecolor='none'
#         )
#         ax.add_patch(rect)
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     dataset = COCODataset("../data/VLM_HD/data/test/img", "../data/VLM_HD/data/test/test.json")
#     visualize(dataset,55)

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.datasets import CocoDetection
from torchvision import transforms
import cv2
from PIL import Image
import math

def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes

class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        applied_transforms = []
        if random.random() < 0.3:
            img, boxes = self.colorJitter(img, boxes)
            applied_transforms.append('colorJitter')
        if random.random() < 0.5:
            img, boxes = self.random_rotation(img, boxes)
            applied_transforms.append('random_rotation')
        if random.random() < 0.5:
            img, boxes = self.random_crop_resize(img, boxes)
            applied_transforms.append('random_crop_resize')

        # if applied_transforms:
        #     # print(f"Applied transforms: {', '.join(applied_transforms)}")
        # else:
        #     # print("No transforms applied")
        return img, boxes

    def colorJitter(self, img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        img = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(img)
        return img, boxes

    def random_rotation(self, img, boxes, degree=10):
        d = random.uniform(-degree, degree)
        w, h = img.size
        rx0, ry0 = w / 2.0, h / 2.0
        img = img.rotate(d)
        a = -d / 180.0 * math.pi
        boxes = torch.from_numpy(boxes)
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, 0] = boxes[:, 1]
        new_boxes[:, 1] = boxes[:, 0]
        new_boxes[:, 2] = boxes[:, 3]
        new_boxes[:, 3] = boxes[:, 2]
        for i in range(boxes.shape[0]):
            ymin, xmin, ymax, xmax = new_boxes[i, :]
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            x0, y0 = xmin, ymin
            x1, y1 = xmin, ymax
            x2, y2 = xmax, ymin
            x3, y3 = xmax, ymax
            z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
            tp = torch.zeros_like(z)
            tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
            tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
            ymax, xmax = torch.max(tp, dim=0)[0]
            ymin, xmin = torch.min(tp, dim=0)[0]
            new_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])
        new_boxes[:, 1::2].clamp_(min=0, max=w - 1)
        new_boxes[:, 0::2].clamp_(min=0, max=h - 1)
        boxes[:, 0] = new_boxes[:, 1]
        boxes[:, 1] = new_boxes[:, 0]
        boxes[:, 2] = new_boxes[:, 3]
        boxes[:, 3] = new_boxes[:, 2]
        boxes = boxes.numpy()
        return img, boxes

    def random_crop_resize(self, img, boxes, crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.7, attempt_max=10):
        success = False
        boxes = torch.from_numpy(boxes)
        for attempt in range(attempt_max):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(crop_scale_min, 1.0) * area
            aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
            w = int(round(math.sqrt(target_area * aspect_ratio_)))
            h = int(round(math.sqrt(target_area / aspect_ratio_)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= img.size[0] and h <= img.size[1]:
                x = random.randint(0, img.size[0] - w)
                y = random.randint(0, img.size[1] - h)
                crop_box = torch.FloatTensor([[x, y, x + w, y + h]])
                inter = _box_inter(crop_box, boxes)
                box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                mask = inter > 0.0001
                inter = inter[mask]
                box_area = box_area[mask.view(-1)]
                box_remain = inter.view(-1) / box_area
                if box_remain.shape[0] != 0:
                    if bool(torch.min(box_remain > remain_min)):
                        success = True
                        break
                else:
                    success = True
                    break
        if success:
            img = img.crop((x, y, x + w, y + h))
            boxes -= torch.Tensor([x, y, x, y])
            boxes[:, 1::2].clamp_(min=0, max=h - 1)
            boxes[:, 0::2].clamp_(min=0, max=w - 1)
        boxes = boxes.numpy()
        return img, boxes

def _box_inter(box1, box2):
    tl = torch.max(box1[:, None, :2], box2[:, :2])  # [n, m, 2]
    br = torch.min(box1[:, None, 2:], box2[:, 2:])  # [n, m, 2]
    hw = (br - tl).clamp(min=0)  # [n, m, 2]
    inter = hw[:, :, 0] * hw[:, :, 1]  # [n, m]
    return inter

class COCODataset(CocoDetection):
    def __init__(self, imgs_path, anno_path, resize_size=[800, 800], is_train=True, transform=None):
        super().__init__(imgs_path, anno_path)
        print("INFO====>check annos, filtering invalid data......")
        ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids = ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.transform = transform
        self.resize_size = resize_size

        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]
        self.train = is_train

    def __getitem__(self, index):
        img, ann = super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]  # xywh --> xyxy

        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.transform is not None:
                img, boxes = self.transform(img, boxes)

        img = np.array(img)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0:
            return False
        if self._has_only_empty_bbox(annot):
            return False
        return True

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
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.0)
            ))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num: max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes

def visualize(dataset, index=None):
    if index is None:
        index = random.randint(0, len(dataset) - 1)

    img, boxes, classes = dataset[index]

    # Convert tensor to numpy array and transpose to (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        if box[0] < 0:
            continue
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    dataset = COCODataset("../data/MTH1000/train/train_images", "../data/MTH1000/train/train.json", transform=Transforms())
    visualize(dataset, 55)
