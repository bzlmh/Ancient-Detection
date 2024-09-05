from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
from dataset.augment import Transforms
import os
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
transform = Transforms()

train_dataset = VOCDataset(root_dir='./data/MTH1200/train', resize_size=[800, 1333],
                           use_difficult=False, is_train=True, augment=transform)
print("train_total_images : {}".format(len(train_dataset)))
eval_dataset = VOCDataset(root_dir='./data/MTH1200/test', resize_size=[800, 1333],
                          use_difficult=False, is_train=False, augment=None)
print("eval_total_images : {}".format(len(eval_dataset)))

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs

model = FCOSDetector().cuda()
model.load_state_dict(torch.load("./checkpoint/model_29.pth"))
model.eval()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                          collate_fn=eval_dataset.collate_fn)

def count_detected_objects(model, data_loader):
    model.eval()
    total_detected_objects = []

    with torch.no_grad():
        for images, _, _ in tqdm(data_loader, desc="Counting Detected Objects", unit="image"):
            images = images.cuda()
            outputs = model(images)
            num_detected_objects = len(outputs[0][0])  # Assuming each batch contains only one image
            total_detected_objects.append(num_detected_objects)

    return total_detected_objects

total_detected_objects = count_detected_objects(model, eval_loader)
for i, num_objects in enumerate(total_detected_objects):
    print("Image {}: Detected {} objects".format(i + 1, num_objects))
