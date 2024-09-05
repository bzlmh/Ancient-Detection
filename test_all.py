from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
from dataset.augment import Transforms
import os
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
from PIL import ImageDraw, Image

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=str, default='0', help="number of gpu to use")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True

transform = Transforms()

eval_dataset = VOCDataset(root_dir='./data/MTH1200/test', resize_size=[800, 1333],
                          use_difficult=False, is_train=False, augment=None)
print("eval_total_images : {}".format(len(eval_dataset)))

eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                          collate_fn=eval_dataset.collate_fn)

model = FCOSDetector().cuda()
model.load_state_dict(torch.load("./checkpoint/model_29.pth"))
model.eval()

def compute_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return None, None
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    matched_true_boxes = set()

    for pred_box in pred_boxes:
        best_iou = 0
        matched_true_box = None
        pred_box = torch.tensor(pred_box)
        for true_box in true_boxes[0]:
            iou = compute_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                matched_true_box = true_box
        if best_iou >= iou_threshold:
            true_positive += 1
            matched_true_boxes.add(tuple(matched_true_box))
        else:
            false_positive += 1

    false_negative = len(true_boxes) - len(matched_true_boxes)
    precision = true_positive / (true_positive + false_positive + 1e-9)
    recall = true_positive / (true_positive + false_negative + 1e-9)

    return precision, recall


def compute_iou(box_a, box_b):
    x1 = torch.max(box_a[0], box_b[0])
    y1 = torch.max(box_a[1], box_b[1])
    x2 = torch.min(box_a[2], box_b[2])
    y2 = torch.min(box_a[3], box_b[3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    iou = intersection / (union + 1e-9)

    return iou


def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def evaluate(model, data_loader, iou_threshold=0.5, save_dir='./eval_results'):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    all_metrics = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                images, true_boxes, img_paths = data
                images = images.cuda()
                outputs = model(images)

                pred_boxes = outputs[2][0].cpu().numpy()
                all_pred_boxes.append(pred_boxes)
                all_true_boxes.append(true_boxes)

                # Save images with predicted bounding boxes
                for img, pred_box in zip(images, pred_boxes):
                    img = img.permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    draw = ImageDraw.Draw(img)
                    x_min = int(round(pred_box[0]))
                    y_min = int(round(pred_box[1]))
                    x_max = int(round(pred_box[2]))
                    y_max = int(round(pred_box[3]))
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red")
                    img_name = "image_{}.png".format(i)  # Example: image_0.png, image_1.png, etc.
                    img.save(os.path.join(save_dir, img_name))

                pbar.update(1)

    for pred_boxes, true_boxes in zip(all_pred_boxes, all_true_boxes):
        precision, recall = compute_precision_recall(pred_boxes, true_boxes, iou_threshold)
        f1_score = compute_f1_score(precision, recall)
        all_metrics.append((precision, recall, f1_score))

    return all_metrics

all_evaluation_metrics = evaluate(model, eval_loader)

total_precision = 0
total_recall = 0
total_f1_score = 0

for precision, recall, f1_score in all_evaluation_metrics:
    total_precision += precision
    total_recall += recall
    total_f1_score += f1_score

avg_precision = total_precision / len(all_evaluation_metrics)
avg_recall = total_recall / len(all_evaluation_metrics)
avg_f1_score = total_f1_score / len(all_evaluation_metrics)

print("Average Precision: {:.4f}, Average Recall: {:.4f}, Average F1 Score: {:.4f}".format(avg_precision, avg_recall, avg_f1_score))