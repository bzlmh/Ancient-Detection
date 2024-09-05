
import cv2
import torch
from model.fcos import FCOSDetector
from torchvision import transforms
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np
def preprocess_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
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
class Config:
    pretrained = False
    freeze_stage_1 = False
    freeze_bn = False
    fpn_out_channels = 256
    use_p5 = True
    class_num = 1
    use_GN_head = True
    prior = 0.01
    add_centerness = True
    cnt_on_reg = True
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
    score_threshold = 0.3
    nms_iou_threshold = 0.1
    max_detection_boxes_num = 2500
    STD_TH = 0.7
    STD_NMS = True  # 不启用标准 NMS
    STD_SOFT = 'soft'  # 启用 Soft NMS
    STD_METHOD = 'stdiou'
    STD_IOU_SIGMA = 0.01
    XYXY = False
    PRED_STD = True
    PRED_STD_LOG = True

cfg = Config()

def compute_iou(fusion_box, boxes):
    """计算框的IoU
    "Weighted Boxes Fusion: ensembling boxes for object detection models"
    https://arxiv.org/abs/1910.13302

    Args:
        fusion_box: (5,)
                    np.array
        boxes:      (n, 5)
                    np.array

    Returns:
        ious

    """
    fusion_box_min = fusion_box[:2]  # 获取融合框的左上角坐标
    fusion_box_max = fusion_box[2:4]  # 获取融合框的右下角坐标
    boxes_min = boxes[..., :2]  # 获取所有框的左上角坐标
    boxes_max = boxes[..., 2:4]  # 获取所有框的右下角坐标
    fusion_wh = fusion_box_max - fusion_box_min  # 计算融合框的宽和高
    boxes_wh = boxes_max - boxes_min  # 计算所有框的宽和高
    fusion_area = fusion_wh[0] * fusion_wh[1]  # 计算融合框的面积
    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]  # 计算所有框的面积

    inter_min = np.maximum(fusion_box_min, boxes_min)  # 计算交集区域的左上角坐标
    inter_max = np.minimum(fusion_box_max, boxes_max)  # 计算交集区域的右下角坐标
    inter_wh = np.maximum(0, inter_max - inter_min)  # 计算交集区域的宽和高
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # 计算交集区域的面积

    ious = inter_area / (fusion_area + boxes_area - inter_area)  # 计算IoU

    return ious

def compute_diou(fusion_box, boxes):
    """计算框的DIoU
    "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    https://arxiv.org/abs/1911.08287

    Args:
        fusion_box: (5,)
                    np.array
        boxes:      (n, 5)
                    np.array

    Returns:
        dious

    """
    fusion_box_min = fusion_box[:2]  # 获取融合框的左上角坐标
    fusion_box_max = fusion_box[2:4]  # 获取融合框的右下角坐标
    boxes_min = boxes[..., :2]  # 获取所有框的左上角坐标
    boxes_max = boxes[..., 2:4]  # 获取所有框的右下角坐标
    fusion_wh = fusion_box_max - fusion_box_min  # 计算融合框的宽和高
    boxes_wh = boxes_max - boxes_min  # 计算所有框的宽和高
    fusion_area = fusion_wh[0] * fusion_wh[1]  # 计算融合框的面积
    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]  # 计算所有框的面积

    inter_min = np.maximum(fusion_box_min, boxes_min)  # 计算交集区域的左上角坐标
    inter_max = np.minimum(fusion_box_max, boxes_max)  # 计算交集区域的右下角坐标
    inter_wh = np.maximum(0, inter_max - inter_min)  # 计算交集区域的宽和高
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # 计算交集区域的面积

    iou = inter_area / (fusion_area + boxes_area - inter_area)  # 计算IoU

    fusion_center = (fusion_box_min + fusion_box_max) / 2  # 计算融合框的中心点
    boxes_center = (boxes_min + boxes_max) / 2  # 计算所有框的中心点
    center_distance = np.sum((fusion_center - boxes_center) ** 2, axis=1)  # 计算中心点之间的距离平方

    enclosing_min = np.minimum(fusion_box_min, boxes_min)  # 计算包含框的左上角坐标
    enclosing_max = np.maximum(fusion_box_max, boxes_max)  # 计算包含框的右下角坐标
    enclosing_wh = enclosing_max - enclosing_min  # 计算包含框的宽和高
    enclosing_diag = np.sum(enclosing_wh ** 2)  # 计算包含框对角线长度的平方

    dious = iou - (center_distance / enclosing_diag)  # 计算DIoU

    return dious
def soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=2):
    """Soft-NMS implementation

    Args:
        boxes: np.array: [[x1, y1, x2, y2, score], ..., [x1, y1, x2, y2, score]]
        sigma: Gaussian penalty sigma
        Nt: IoU threshold
        threshold: score threshold to discard boxes
        method: 1 for linear, 2 for Gaussian, 3 for original NMS

    Returns:
        boxes: np.array: remaining boxes after Soft-NMS
    """
    N = boxes.shape[0]
    for i in range(N):
        maxpos = i + np.argmax(boxes[i:, 4])
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j].reshape(1, -1))
            iou = float(iou)  # 将iou转换为标量
            if method == 1:  # linear
                if iou > Nt:
                    boxes[j, 4] = boxes[j, 4] * (1 - iou)
            elif method == 2:  # Gaussian
                boxes[j, 4] = boxes[j, 4] * np.exp(-(iou ** 2) / sigma)
            elif method == 3:  # original NMS
                if iou > Nt:
                    boxes[j, 4] = 0
        boxes = boxes[boxes[:, 4] > threshold]
    return boxes
def weighted_boxes_fusion(boxes_list, iou_thres, score_thres):
    """对每个类别进行加权框融合

    Args:
        boxes_list:  list of np.array: [[x1, y1, x2, y2, score], ..., [x1, y1, x2, y2, score]]
                     每个元素是来自同一类别的框
        iou_thres:   IoU阈值
        score_thres: 置信度阈值

    Returns:
        results: 融合后的框
                 [x1, y1, x2, y2, score]

    """
    F = []  # 存储融合后的框
    L = []  # 存储每个融合框对应的原始框索引列表

    all_boxes = np.concatenate(boxes_list, axis=0)  # 合并来自所有模型的检测框
    scores = all_boxes[..., 4]  # 获取所有框的置信度分数
    valid_index = np.where(scores > score_thres)[0]  # 获取置信度高于阈值的框的索引
    valid_scores = scores[valid_index]  # 获取有效框的置信度分数
    index_sorted = np.argsort(valid_scores)[::-1]  # 对置信度分数进行降序排序
    boxes_sorted = all_boxes[valid_index][index_sorted]  # 根据排序结果获取排序后的框

    if len(boxes_sorted) > 0:
        selected_index = []  # 存储已选择的框的索引
        if len(F) == 0:
            F.append(boxes_sorted[0])  # 将第一个框添加到融合框列表中
            L.append([valid_index[index_sorted[0]]])  # 添加第一个框的索引到L中
            selected_index.append(valid_index[index_sorted[0]])  # 标记第一个框已被选择

        remain_index = set(valid_index[index_sorted]) - set(selected_index)  # 剩余框的索引列表

        while len(remain_index) > 0:  # 当还有未处理的框时
            num_fusion = len(F)  # 当前融合框的数量
            index = np.array(list(remain_index))  # 将剩余索引转换为数组
            remain_boxes = all_boxes[index]  # 获取剩余的框
            ious = compute_diou(F[num_fusion - 1], remain_boxes)  # 计算当前融合框与剩余框的IoU
            matched_index = np.where(ious > iou_thres)[0]  # 获取IoU大于阈值的框的索引
            if len(matched_index) > 0:
                matched_index = index[matched_index]
                L[num_fusion - 1].extend(matched_index)  # 将匹配的框添加到L中的对应位置
                remain_index = remain_index - set(matched_index)
            else:
                F.append(all_boxes[list(remain_index)[0]])  # 将当前剩余框中的第一个框作为新的融合框添加到F和L中
                L.append([list(remain_index)[0]])
                remain_index = remain_index - {list(remain_index)[0]}

    results = []
    for i in range(len(F)):
        fusion_boxes = all_boxes[np.array(L[i]).astype('int32')]  # 获取当前融合框对应的所有原始框
        scores_ = fusion_boxes[..., -1]  # 获取这些框的置信度分数
        location = fusion_boxes[..., :4]  # 获取这些框的位置
        score_max = np.array([np.max(scores_)])  # 获取最高的置信度分数
        weighted_location = location * scores_[:, np.newaxis]  # 计算加权位置
        weighted_location_sum = np.sum(weighted_location, axis=0)  # 加权位置的和
        scores_sum = np.sum(scores_)  # 置信度分数的和
        results_location = weighted_location_sum / scores_sum  # 计算最终的加权位置
        results.append(np.concatenate([results_location, score_max]))  # 将结果添加到结果列表中

    return np.array(results)  # 返回结果


if __name__ == "__main__":
    model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/model_600.pth", map_location=torch.device('cpu')))
    model = model.eval()
    model = convertSyncBNtoBN(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("===>success loading model")

    origin_root = "./data//MTH1000/test/test_images/"
    binary_root = "./data//MTH1000/test/b_img/"
    names = os.listdir(origin_root)

    target_size = (800, 800)  # 统一训练尺寸
    for name in names:
        # 处理原始图像
        origin_img_bgr = cv2.imread(origin_root + name)
        origin_img_h, origin_img_w = origin_img_bgr.shape[:2]

        # 调整原始图像尺寸为800x800
        resized_origin_img = cv2.resize(origin_img_bgr, target_size)
        origin_image = preprocess_img(resized_origin_img).to(device)

        start_t = time.time()
        with torch.no_grad():
            origin_out = model(origin_image.unsqueeze(dim=0))
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing original image, cost time %.2f ms" % cost_t)

        origin_scores, origin_classes, origin_boxes = origin_out
        origin_boxes = origin_boxes[0].cpu().numpy()
        origin_classes = origin_classes[0].cpu().numpy().tolist()
        origin_scores = origin_scores[0].cpu().numpy().tolist()

        all_origin_boxes = []
        for i, box in enumerate(origin_boxes):
            if origin_scores[i] < Config.score_threshold:
                continue
            adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), origin_scores[i]]
            all_origin_boxes.append(adjusted_box)

        # 处理二值化图像
        binary_img_bgr = cv2.imread(binary_root + name)
        binary_img_h, binary_img_w = binary_img_bgr.shape[:2]

        # 调整二值化图像尺寸为800x800
        resized_binary_img = cv2.resize(binary_img_bgr, target_size)
        binary_image = preprocess_img(resized_binary_img).to(device)

        start_t = time.time()
        with torch.no_grad():
            binary_out = model(binary_image.unsqueeze(dim=0))
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing binary image, cost time %.2f ms" % cost_t)

        binary_scores, binary_classes, binary_boxes = binary_out
        binary_boxes = binary_boxes[0].cpu().numpy()
        binary_classes = binary_classes[0].cpu().numpy().tolist()
        binary_scores = binary_scores[0].cpu().numpy().tolist()

        all_binary_boxes = []
        for i, box in enumerate(binary_boxes):
            if binary_scores[i] < Config.score_threshold:
                continue
            adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), binary_scores[i]]
            all_binary_boxes.append(adjusted_box)

        # 合并结果
        merged_results = np.vstack((all_origin_boxes, all_binary_boxes))

        # 使用 soft NMS 方法
        merged_boxes_list = [soft_nms(np.array(merged_results), sigma=0.5, Nt=0.3, threshold=0.001)]
        merged_results = weighted_boxes_fusion(merged_boxes_list, 0.55, 0.05)

        filtered_boxes = merged_results[:, :4].astype(int)

        # 将预测框调整回原始图像尺寸
        original_scale = (origin_img_w / target_size[1], origin_img_h / target_size[0])
        adjusted_boxes = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * original_scale[0])
            y1 = int(y1 * original_scale[1])
            x2 = int(x2 * original_scale[0])
            y2 = int(y2 * original_scale[1])
            adjusted_boxes.append([x1, y1, x2, y2])

        adjusted_boxes = np.array(adjusted_boxes)

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(origin_img_bgr, cv2.COLOR_BGR2RGB))

        for box in adjusted_boxes:
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            b_color = 'green'
            bbox = patches.Rectangle((pt1[0], pt1[1]), width=pt2[0] - pt1[0], height=pt2[1] - pt1[1], linewidth=0.5,
                                     facecolor='none', edgecolor=b_color)
            ax.add_patch(bbox)

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        plt.savefig(f'results/our/{name}', dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close()

# if __name__ == "__main__":
#     model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
#     model = torch.nn.DataParallel(model)
#     model.load_state_dict(torch.load("./checkpoint/model_30.pth", map_location=torch.device('cpu')))
#     model = model.eval()
#     model = convertSyncBNtoBN(model)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     print("===>success loading model")
#
#     origin_root = "./data/competition/origin/"
#     binary_root= "./data/competition/mask/"
#     names = os.listdir(origin_root)
#     for name in names:
#         # 处理原始图像
#         origin_img_bgr = cv2.imread(origin_root + name)
#         origin_img_h, origin_img_w = origin_img_bgr.shape[:2]
#
#         origin_image = preprocess_img(origin_img_bgr).to(device)
#
#         start_t = time.time()
#         with torch.no_grad():
#             origin_out = model(origin_image.unsqueeze(dim=0))
#         end_t = time.time()
#         cost_t = 1000 * (end_t - start_t)
#         print("===>success processing original image, cost time %.2f ms" % cost_t)
#
#         origin_scores, origin_classes, origin_boxes = origin_out
#         origin_boxes = origin_boxes[0].cpu().numpy()
#         origin_classes = origin_classes[0].cpu().numpy().tolist()
#         origin_scores = origin_scores[0].cpu().numpy().tolist()
#
#         all_origin_boxes = []
#         for i, box in enumerate(origin_boxes):
#             if origin_scores[i] < Config.score_threshold:
#                 continue
#             adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), origin_scores[i]]
#             all_origin_boxes.append(adjusted_box)
#
#         # 处理二值化图像
#         binary_img_bgr = cv2.imread(binary_root + name)
#         binary_img_h, binary_img_w = binary_img_bgr.shape[:2]
#
#         binary_image = preprocess_img(binary_img_bgr).to(device)
#
#         start_t = time.time()
#         with torch.no_grad():
#             binary_out = model(binary_image.unsqueeze(dim=0))
#         end_t = time.time()
#         cost_t = 1000 * (end_t - start_t)
#         print("===>success processing binary image, cost time %.2f ms" % cost_t)
#
#         binary_scores, binary_classes, binary_boxes = binary_out
#         binary_boxes = binary_boxes[0].cpu().numpy()
#         binary_classes = binary_classes[0].cpu().numpy().tolist()
#         binary_scores = binary_scores[0].cpu().numpy().tolist()
#
#         all_binary_boxes = []
#         for i, box in enumerate(binary_boxes):
#             if binary_scores[i] < Config.score_threshold:
#                 continue
#             adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), binary_scores[i]]
#             all_binary_boxes.append(adjusted_box)
#
#         # 合并结果
#         merged_results = np.vstack((all_origin_boxes, all_binary_boxes))
#
#         # 使用 soft NMS 方法
#         merged_boxes_list = [soft_nms(np.array(merged_results), sigma=0.5, Nt=0.3, threshold=0.001)]
#         merged_results = weighted_boxes_fusion(merged_boxes_list, 0.55, 0.05)
#
#         filtered_boxes = merged_results[:, :4].astype(int)
#
#         plt.figure()
#         fig, ax = plt.subplots(1)
#         ax.imshow(cv2.cvtColor(origin_img_bgr, cv2.COLOR_BGR2RGB))
#
#         for box in filtered_boxes:
#             pt1 = (box[0], box[1])
#             pt2 = (box[2], box[3])
#             b_color = 'green'
#             bbox = patches.Rectangle((pt1[0], pt1[1]), width=pt2[0] - pt1[0], height=pt2[1] - pt1[1], linewidth=0.5,
#                                      facecolor='none', edgecolor=b_color)
#             ax.add_patch(bbox)
#
#         plt.axis('off')
#         plt.gca().xaxis.set_major_locator(NullLocator())
#         plt.gca().yaxis.set_major_locator(NullLocator())
#
#         plt.savefig(f'results/our/{name}', dpi=300, bbox_inches='tight', pad_inches=0.0)
#         plt.close()










# if __name__ == "__main__":
#     model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
#     model = torch.nn.DataParallel(model)
#     model.load_state_dict(torch.load("./checkpoint/model_109.pth", map_location=torch.device('cpu')))
#     model = model.eval()
#     model = convertSyncBNtoBN(model)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     print("===>success loading model")
#
#     import os
#
#     origin_root = "./data/Degard/test/ostu/"
#     names = os.listdir(origin_root)
#     for name in names:
#         img_bgr = cv2.imread(origin_root + name)
#         img_h, img_w = img_bgr.shape[:2]
#
#         image = preprocess_img(img_bgr).to(device)
#
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
#         for i, box in enumerate(boxes):
#             if scores[i] < Config.score_threshold:
#                 continue
#             adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), scores[i]]  # 在盒子坐标后添加分数
#             all_boxes.append(adjusted_box)
#         # 使用 soft NMS 方法
#         boxes_list = [soft_nms(np.array(all_boxes), sigma=0.5, Nt=0.3, threshold=0.001)]
#         results_ = weighted_boxes_fusion(boxes_list,0.55, 0.05)  # 使用WBF融合过滤后的检测结果
#
#         filtered_boxes = results_[:, :4].astype(int)
#
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
#         plt.savefig(f'results/otsu/{name}', dpi=300, bbox_inches='tight', pad_inches=0.0)
#         plt.close()
