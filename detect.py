#------------------------init NMS------------------------
import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def preprocess_img(image):
    return image

def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name, convertSyncBNtoBN(child))
    del module
    return module_output

if __name__ == "__main__":
    cmap = plt.get_cmap('tab20b')
    colors = ['red'] * 10

    class Config():
        pretrained = False
        freeze_stage_1 = False
        freeze_bn = False
        fpn_out_channels = 256
        use_p5 = True
        class_num = 1
        use_GN_head = True
        prior = 0.01
        add_centerness = True
        cnt_on_reg = False
        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        score_threshold = 0.2
        nms_iou_threshold = 0.2
        max_detection_boxes_num = 300

    model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/model_32.pth", map_location=torch.device('cpu')))
    model = model.eval()
    model = convertSyncBNtoBN(model)  # 转换SyncBatchNorm层为BatchNorm层

    # 将模型移动到GPU上（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("===>success loading model")

    import os
    root = "./data/VLM_HD/data/test/img/"
    names = os.listdir(root)
    for name in names:
        img_bgr = cv2.imread(root + name)
        img = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        img1 = transforms.ToTensor()(img)
        img1 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)(img1)
        img1 = img1.to(device)  # 将输入图像移动到GPU上

        start_t = time.time()
        with torch.no_grad():
            out = model(img1.unsqueeze_(dim=0))
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)

        scores, classes, boxes = out
        boxes = boxes[0].cpu().numpy()
        print("Detected boxes:", boxes)

        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            b_color = colors[int(classes[i]) - 1]
            bbox = patches.Rectangle((box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], linewidth=1,
                                     facecolor='none', edgecolor=b_color)
            ax.add_patch(bbox)

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        plt.savefig('out_images/{}'.format(name), dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close()
# # #------------------------Soft NMS------------------------
# import cv2
# import torch
# from model.fcos import FCOSDetector
# from torchvision import transforms
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.ticker import NullLocator
#
# class Config:
#     pretrained = False
#     freeze_stage_1 = False
#     freeze_bn = False
#     fpn_out_channels = 256
#     use_p5 = True
#     class_num = 1
#     use_GN_head = True
#     prior = 0.01
#     add_centerness = True
#     cnt_on_reg = True
#     strides = [8, 16, 32, 64, 128]
#     limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
#     score_threshold = 0.3
#     nms_iou_threshold = 0.1
#     max_detection_boxes_num = 2500
#     STD_TH = 0.7
#     STD_NMS = True  # 不启用标准 NMS
#     STD_SOFT = 'soft'  # 启用 Soft NMS
#     STD_METHOD = 'stdiou'
#     STD_IOU_SIGMA = 0.01
#     XYXY = False
#     PRED_STD = True
#     PRED_STD_LOG = True
#
# cfg = Config()
#
# def preprocess_img(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = transforms.ToTensor()(image)
#     image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
#     return image
#
# def convertSyncBNtoBN(module):
#     module_output = module
#     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#         module_output = torch.nn.BatchNorm2d(module.num_features,
#                                              module.eps, module.momentum,
#                                              module.affine,
#                                              module.track_running_stats)
#         if module.affine:
#             module_output.weight.data = module.weight.data.clone().detach()
#             module_output.bias.data = module.bias.data.clone().detach()
#         module_output.running_mean = module.running_mean
#         module_output.running_var = module.running_var
#     for name, child in module.named_children():
#         module_output.add_module(name, convertSyncBNtoBN(child))
#     del module
#     return module_output
#
#
# def soft(dets, confidence=None, ax=None, max_boxes=2800):
#     thresh = cfg.STD_TH
#     if cfg.STD_METHOD == 'stdiou' and thresh > .1:
#         thresh = 0.01
#     sigma = .5
#     N = len(dets)
#     x1 = dets[:, 0].copy()
#     y1 = dets[:, 1].copy()
#     x2 = dets[:, 2].copy()
#     y2 = dets[:, 3].copy()
#     scores = dets[:, 4].copy()
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     ious = np.zeros((N, N))
#
#     # 计算IOU
#     for i in range(N):
#         xx1 = np.maximum(x1[i], x1)
#         yy1 = np.maximum(y1[i], y1)
#         xx2 = np.minimum(x2[i], x2)
#         yy2 = np.minimum(y2[i], y2)
#
#         w = np.maximum(0.0, xx2 - xx1 + 1.)
#         h = np.maximum(0.0, yy2 - yy1 + 1.)
#         inter = w * h
#         ovr = inter / (areas[i] + areas - inter)
#         ious[i, :] = ovr.copy()
#
#     i = 0
#     while i < N:
#         maxpos = dets[i:N, 4].argmax()
#         maxpos += i
#         dets[[maxpos, i]] = dets[[i, maxpos]]
#         confidence[[maxpos, i]] = confidence[[i, maxpos]]
#         ious[[maxpos, i]] = ious[[i, maxpos]]
#         ious[:, [maxpos, i]] = ious[:, [i, maxpos]]
#
#         ovr_bbox = np.where((ious[i, i:N] > thresh))[0] + i
#         if cfg.STD_METHOD == 'stdiou':
#             p = np.exp(-(1 - ious[i, ovr_bbox]) ** 2 / cfg.STD_IOU_SIGMA)
#             p = p.reshape(-1, 1)
#             dets[i, :4] = np.sum(p * dets[ovr_bbox, :4] / (confidence[ovr_bbox].reshape(-1, 1) ** 2), axis=0) / np.sum(
#                 p / (confidence[ovr_bbox].reshape(-1, 1) ** 2), axis=0)
#         else:
#             assert cfg.STD_METHOD == 'soft'
#
#         # 合并高IoU的框
#         high_iou_indices = np.where(ious[i, :] > 0.3)[0]  # 设定一个高IoU阈值，比如0.5
#         if len(high_iou_indices) > 1:
#             x1_comb = np.min(dets[high_iou_indices, 0])
#             y1_comb = np.min(dets[high_iou_indices, 1])
#             x2_comb = np.max(dets[high_iou_indices, 2])
#             y2_comb = np.max(dets[high_iou_indices, 3])
#             score_comb = np.mean(dets[high_iou_indices, 4])
#             dets[i, :4] = [x1_comb, y1_comb, x2_comb, y2_comb]
#             dets[i, 4] = score_comb
#             confidence[i] = score_comb
#             # 将其他高IoU的框的得分置为0，以便在后续过程中去除它们
#             for idx in high_iou_indices:
#                 if idx != i:
#                     dets[idx, 4] = 0
#
#         pos = i + 1
#         while pos < N:
#             if ious[i, pos] > 0:
#                 ovr = ious[i, pos]
#                 if cfg.STD_SOFT == 'hard':
#                     if ious[i, pos] > cfg.TEST.NMS:
#                         dets[pos, 4] = 0
#                 else:
#                     dets[pos, 4] *= np.exp(-(ovr * ovr) / sigma)
#                 if dets[pos, 4] < 0.001:
#                     dets[[pos, N - 1]] = dets[[N - 1, pos]]
#                     confidence[[pos, N - 1]] = confidence[[N - 1, pos]]
#                     ious[[pos, N - 1]] = ious[[N - 1, pos]]
#                     ious[:, [pos, N - 1]] = ious[:, [N - 1, pos]]
#                     N -= 1
#                     pos -= 1
#             pos += 1
#         i += 1
#
#     # 按置信度排序并限制检测框数量
#     keep = [i for i in range(N)]
#     dets = dets[keep]
#     dets = dets[dets[:, 4].argsort()[::-1]]  # 按置信度从高到低排序
#     if len(dets) > max_boxes:
#         dets = dets[:max_boxes]
#     keep = [i for i in range(len(dets))]
#     return dets, keep
#
# import xml.etree.ElementTree as ET
# from xml.dom import minidom
#
# def create_pascal_voc_xml(filename, width, height, detections):
#     """
#     创建 Pascal VOC 格式的 XML 文件
#     :param filename: 图像文件名
#     :param width: 图像宽度
#     :param height: 图像高度
#     :param detections: 检测结果，格式为 [(xmin, ymin, xmax, ymax, score)]
#     :return: XML 格式的字符串
#     """
#     annotation = ET.Element("annotation")
#     ET.SubElement(annotation, "folder").text = "images"
#     ET.SubElement(annotation, "filename").text = filename
#
#     size = ET.SubElement(annotation, "size")
#     ET.SubElement(size, "width").text = str(width)
#     ET.SubElement(size, "height").text = str(height)
#     ET.SubElement(size, "depth").text = "3"
#
#     for (xmin, ymin, xmax, ymax, score) in detections:
#         obj = ET.SubElement(annotation, "object")
#         ET.SubElement(obj, "name").text = "char"
#         ET.SubElement(obj, "pose").text = "Unspecified"
#         ET.SubElement(obj, "truncated").text = "0"
#         ET.SubElement(obj, "difficult").text = "0"
#
#         bndbox = ET.SubElement(obj, "bndbox")
#         ET.SubElement(bndbox, "xmin").text = str(xmin)
#         ET.SubElement(bndbox, "ymin").text = str(ymin)
#         ET.SubElement(bndbox, "xmax").text = str(xmax)
#         ET.SubElement(bndbox, "ymax").text = str(ymax)
#
#     rough_string = ET.tostring(annotation, 'utf-8')
#     reparsed = minidom.parseString(rough_string)
#     return reparsed.toprettyxml(indent="  ")
#
# def save_detection_to_xml(filename, width, height, detections, save_path):
#     """
#     将检测结果保存为 XML 文件
#     :param filename: 图像文件名
#     :param width: 图像宽度
#     :param height: 图像高度
#     :param detections: 检测结果，格式为 [(xmin, ymin, xmax, ymax, score)]
#     :param save_path: 保存 XML 文件的路径
#     """
#     xml_str = create_pascal_voc_xml(filename, width, height, detections)
#     with open(save_path, "w") as f:
#         f.write(xml_str)
#     print(f"Saved {save_path}")
#
#
# if __name__ == "__main__":
#     model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
#     model = torch.nn.DataParallel(model)
#     model.load_state_dict(torch.load("./checkpoint/model_30.pth", map_location=torch.device('cpu')))
#     model = model.eval()
#     model = convertSyncBNtoBN(model)  # 转换SyncBatchNorm层为BatchNorm层
#
#     # 将模型移动到GPU上（如果可用）
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     print("===>success loading model")
#
#     import os
#     root = "./data/competition/origin/"
#     save_root = "./results/xml/"  # XML文件的保存路径
#     os.makedirs(save_root, exist_ok=True)
#
#     names = os.listdir(root)
#     for name in names:
#         img_bgr = cv2.imread(root + name)
#         img_h, img_w = img_bgr.shape[:2]
#
#         # 将整个图像进行预处理
#         image = preprocess_img(img_bgr).to(device)
#
#         # 模型推断
#         start_t = time.time()
#         with torch.no_grad():
#             out = model(image.unsqueeze(dim=0))
#         end_t = time.time()
#         cost_t = 1000 * (end_t - start_t)
#         print("===>success processing image, cost time %.2f ms" % cost_t)
#
#         scores, classes, boxes = out
#         boxes = boxes[0].cpu().numpy()
#         classes = classes[0].cpu().numpy().tolist()
#         scores = scores[0].cpu().numpy().tolist()
#
#         all_boxes = []
#         all_scores = []
#
#         # 处理模型输出结果
#         for i, box in enumerate(boxes):
#             if scores[i] < Config.score_threshold:
#                 continue
#             adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
#             all_boxes.append(adjusted_box)
#             all_scores.append(scores[i])
#
#         # 使用 soft NMS 方法
#         dets = np.array([all_boxes[i] + [all_scores[i]] for i in range(len(all_boxes))])
#         print(dets)
#         filtered_dets, keep = soft(dets, confidence=np.array(all_scores))
#         filtered_boxes = filtered_dets[:, :4].astype(int)
#
#         # 生成并保存XML文件
#         detections = [(box[0], box[1], box[2], box[3], filtered_dets[i, 4]) for i, box in enumerate(filtered_boxes)]
#         xml_save_path = os.path.join(save_root, os.path.splitext(name)[0] + ".xml")
#         save_detection_to_xml(name, img_w, img_h, detections, xml_save_path)
#
#         # 绘制检测结果
#         plt.figure()
#         fig, ax = plt.subplots(1)
#         ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#
#         for box in filtered_boxes:
#             pt1 = (box[0], box[1])
#             pt2 = (box[2], box[3])
#             b_color = 'green'  # 浅红色
#             bbox = patches.Rectangle((pt1[0], pt1[1]), width=pt2[0] - pt1[0], height=pt2[1] - pt1[1], linewidth=0.5,
#                                      facecolor='none', edgecolor=b_color)
#             ax.add_patch(bbox)
#
#         plt.axis('off')
#         plt.gca().xaxis.set_major_locator(NullLocator())
#         plt.gca().yaxis.set_major_locator(NullLocator())
#
#         plt.savefig(f'results/origin/{name}', dpi=300, bbox_inches='tight', pad_inches=0.0)
#         plt.close()
# # if __name__ == "__main__":
# #     model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
# #     model = torch.nn.DataParallel(model)
# #     model.load_state_dict(torch.load("./checkpoint/model_30.pth", map_location=torch.device('cpu')))
# #     model = model.eval()
# #     model = convertSyncBNtoBN(model)  # 转换SyncBatchNorm层为BatchNorm层
# #
# #     # 将模型移动到GPU上（如果可用）
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     model = model.to(device)
# #     print("===>success loading model")
# #
# #     import os
# #     root = "./data/competition/origin/"
# #     names = os.listdir(root)
# #     for name in names:
# #         img_bgr = cv2.imread(root + name)
# #         img_h, img_w = img_bgr.shape[:2]
# #
# #         # 将整个图像进行预处理
# #         image = preprocess_img(img_bgr).to(device)
# #
# #         # 模型推断
# #         start_t = time.time()
# #         with torch.no_grad():
# #             out = model(image.unsqueeze(dim=0))
# #         end_t = time.time()
# #         cost_t = 1000 * (end_t - start_t)
# #         print("===>success processing image, cost time %.2f ms" % cost_t)
# #
# #         scores, classes, boxes = out
# #         boxes = boxes[0].cpu().numpy()
# #         classes = classes[0].cpu().numpy().tolist()
# #         scores = scores[0].cpu().numpy().tolist()
# #
# #         all_boxes = []
# #         all_scores = []
# #
# #         # 处理模型输出结果
# #         for i, box in enumerate(boxes):
# #             if scores[i] < Config.score_threshold:
# #                 continue
# #             adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
# #             all_boxes.append(adjusted_box)
# #             all_scores.append(scores[i])
# #
# #         # 使用 soft NMS 方法
# #         dets = np.array([all_boxes[i] + [all_scores[i]] for i in range(len(all_boxes))])
# #         print(dets)
# #         filtered_dets, keep = soft(dets, confidence=np.array(all_scores))
# #         filtered_boxes = filtered_dets[:, :4].astype(int)
# #
# #         plt.figure()
# #         fig, ax = plt.subplots(1)
# #         ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
# #
# #         for box in filtered_boxes:
# #             pt1 = (box[0], box[1])
# #             pt2 = (box[2], box[3])
# #             b_color = 'green'  # 浅红色
# #             bbox = patches.Rectangle((pt1[0], pt1[1]), width=pt2[0] - pt1[0], height=pt2[1] - pt1[1], linewidth=0.5,
# #                                      facecolor='none', edgecolor=b_color)
# #             ax.add_patch(bbox)
# #
# #         plt.axis('off')
# #         plt.gca().xaxis.set_major_locator(NullLocator())
# #         plt.gca().yaxis.set_major_locator(NullLocator())
# #
# #         plt.savefig(f'results/origin/{name}', dpi=300, bbox_inches='tight', pad_inches=0.0)
# #         plt.close()
