from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision import transforms
import cv2
from model.fcos import FCOSDetector
import torch
import os

class COCOGenerator(CocoDetection):
    def __init__(self, imgs_path, anno_path, resize_size=[640, 640]):
        super().__init__(imgs_path, anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids = ids
        self.resize_size = resize_size
        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]

    def __getitem__(self, index):
        img, ann = super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        img = np.array(img)

        img, boxes, scale = self.preprocess_img_boxes(img, boxes, self.resize_size)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.mean, self.std, inplace=True)(img)

        return img, boxes, scale

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
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
            return image_paded, boxes, scale

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True


def evaluate_single_image(generator, model, image_index, threshold=0.05, save_path="predicted_images"):
    """ Use the model to evaluate a single image and save the predicted image with bounding boxes.

    Args:
        generator : The dataset generator.
        model     : The model to evaluate.
        image_index : Index of the image in the dataset to evaluate.
        threshold : The score threshold to use.
        save_path : The directory to save predicted images.
    """
    img, gt_boxes, scale = generator[image_index]
    img_path = generator.coco.loadImgs(generator.ids[image_index])[0]['file_name']

    # Run network
    scores, _, boxes = model(img.unsqueeze(dim=0).cuda())
    scores = scores.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    boxes /= scale

    # Correct boxes for image scale
    # Change to (x, y, w, h) (MS COCO standard)
    boxes[:, :, 2] -= boxes[:, :, 0]
    boxes[:, :, 3] -= boxes[:, :, 1]

    # Create a canvas to draw the bounding boxes
    img_with_boxes = img.permute(1, 2, 0).numpy().copy()
    img_with_boxes *= np.array(generator.std)  # Undo normalization
    img_with_boxes += np.array(generator.mean)
    img_with_boxes = (img_with_boxes * 255).astype(np.uint8)

    # Prepare detections for evaluation
    detections = []
    for box, score in zip(boxes[0], scores[0]):
        # Scores are sorted, so we can break
        if score < threshold:
            break

        box = [int(coord) for coord in box]
        box_annotation = {'image_id': generator.ids[image_index], 'score': float(score), 'bbox': box, 'category_id': 0}
        detections.append(box_annotation)


        cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    # Save the image with bounding boxes
    img_name = f"predicted_{generator.ids[image_index]}.jpg"
    cv2.imwrite(os.path.join(save_path, img_name), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

    return detections


def evaluate_coco_single_image(generator, model, image_index, threshold=0.05):
    """ Use the model to evaluate a single image and print evaluation results.

    Args:
        generator : The dataset generator.
        model     : The model to evaluate.
        image_index : Index of the image in the dataset to evaluate.
        threshold : The score threshold to use.
    """
    detections = evaluate_single_image(generator, model, image_index, threshold)

    # Load ground truth annotations
    coco_gt = generator.coco

    # Load predicted annotations
    coco_dt = coco_gt.loadRes(detections)

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    img_id = generator.ids[image_index]
    img_name = generator.coco.loadImgs(img_id)[0]['file_name']

    print(f"Image: {img_name}")
    print(f"Detected {coco_eval.stats[0]} objects")
    print(f"Average number of detections per image: {coco_eval.stats[1]}")
    print(f"Detection Precision: {coco_eval.stats[8]}")
    print(f"Detection Recall: {coco_eval.stats[9]}")

    return coco_eval.stats

if __name__ == "__main__":
    generator = COCOGenerator("./data/Degard/train/img", "./data/Degard/train/train.json")
    model = FCOSDetector(mode="inference")
    model = torch.nn.DataParallel(model)
    model = model.cuda().eval()
    model.load_state_dict(torch.load("./checkpoint/model_100.pth", map_location=torch.device('cuda')))
    
    # Choose an image index to evaluate
    image_index = 0
    
    # Evaluate the single image
    evaluate_coco_single_image(generator, model, image_index)
