from detector import Detector
import cv2 as cv
import torch
from torch.autograd import Variable
import os
import sys
import time
import datetime
import argparse
import tqdm
import numpy as np
from utils.utils import *
from datasets.fot_seq_for_YOLO import get_loader
from utils.visualization import plot_detections
from model.darknet import Darknet


model = Darknet('../config/yolov3.cfg', img_size=608)
model.load_weights('../weights/yolov3.weights')
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
model.eval()
dataloader = get_loader(root='data/football', name='5', min_height=0, num_workers=0, batch_size=2)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

all_detections = []
all_annotations = []

for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

    imgs = Variable(imgs.type(Tensor))
    with torch.no_grad():
        detections = model(imgs)
        outputs = non_max_suppression(detections, num_classes=80, conf_thres=0.5, nms_thres=0.4)

    for output, annotations in zip(outputs, targets):
        all_detections.append([np.array([]) for _ in range(1)])

        if output is not None:
            # Get predicted boxes, confidence scores and labels
            pred_boxes = output[:, :5].cpu().numpy()
            scores = output[:, 4].cpu().numpy()
            pred_labels = output[:, -1].cpu().numpy()

            # Order by confidence
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            # print('pred_boxes', pred_boxes)
            for label in range(1):
                all_detections[-1][label] = pred_boxes[pred_labels == label]

        all_annotations.append([np.array([]) for _ in range(1)])
        if any(annotations[:, -1] > 0):

            annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

            # Reformat to x1, y1, x2, y2 and rescale to image dimensions
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
            annotation_boxes *= 608

            for label in range(1):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]
average_precisions = {}
for label in range(1):
    true_positives = []
    scores = []
    num_annotations = 0

    for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]

        num_annotations += annotations.shape[0]
        detected_annotations = []
        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.append(0)
                continue

            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= 0.5 and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # no annotations -> AP for this class is 0
    if num_annotations == 0:
        average_precisions[label] = 0
        continue

    true_positives = np.array(true_positives)
    false_positives = np.ones_like(true_positives) - true_positives
    # sort by score
    indices = np.argsort(-np.array(scores))
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision

print("Average Precisions:")
for c, ap in average_precisions.items():
    print(f"+ Class '{c}' - AP: {ap}")

mAP = np.mean(list(average_precisions.values()))
print(f"mAP: {mAP}")