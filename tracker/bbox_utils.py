import numpy as np
import cv2

import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import *


# [x_min y_min w h] to [x_min y_min x_max y_max]
def xywh_to_xyxy(bboxes):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    xyxy_bboxes = np.zeros_like(bboxes)
    for i in range(len(bboxes)):
        x = bboxes[i][0]
        y = bboxes[i][1]
        w = bboxes[i][2]
        h = bboxes[i][3]
        xyxy_bboxes[i][0] = x
        xyxy_bboxes[i][1] = y
        xyxy_bboxes[i][2] = x + w
        xyxy_bboxes[i][3] = y + h

    return xyxy_bboxes

# [x_min y_min w h] to [x_center y_center w h]
def xywh_xymin_to_xycenter(box):
    return [ int(box[0]+box[2]/2), int(box[1]+box[3]/2), box[2], box[3] ]

def choose_best_box(boxes, last_box):
    """ Choose the box that has highest IOU with last detected box
    """
    if len(boxes) != 1:
        print('{} boxes are found (not only 1 box founded or no boxes founded)'.format(len(boxes)))

    if len(boxes) == 0:
        return last_box
    else:
        best_box = boxes[0]

    max_iou = 0.0
    for box in boxes:
        # box.print_box()
        if bbox_iou(box, last_box) > max_iou:
            max_iou = bbox_iou(box, last_box)
            best_box = box

    if max_iou == 0.0:
        best_box = last_box

    # best_box.print_box()

    return best_box

def draw_normalized_box(image, box):
    xmin  = int((box.x - box.w/2) * image.shape[1])
    xmax  = int((box.x + box.w/2) * image.shape[1])
    ymin  = int((box.y - box.h/2) * image.shape[0])
    ymax  = int((box.y + box.h/2) * image.shape[0])

    # box.print_box()
    # print(xmin, xmax, ymin, ymax)
    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
    # cv2.rectangle(image, (810,165), (860,276), (0,255,0), 3)

    return image

def draw_box(image, box, color=(0,255,0)):
    """ Draw box by [xcenter, ycenter, w, h]
    """
    if isinstance(box, BoundBox):
        xmin = int(box.x - box.w/2)
        xmax = int(box.x + box.w/2)
        ymin = int(box.y - box.h/2)
        ymax = int(box.y + box.h/2)
    else:
        xmin  = int(box[0] - box[2]/2)
        xmax  = int(box[0] + box[2]/2)
        ymin  = int(box[1] - box[3]/2)
        ymax  = int(box[1] + box[3]/2)
    
    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 3)

    return image

def normalize_box(image_shape, unnormailzed_box):
    """ Normalize box label into [0,1] by image size
    """
    if isinstance(unnormailzed_box, BoundBox):
        x = unnormailzed_box.x / image_shape[1]
        w = unnormailzed_box.w / image_shape[1]
        y = unnormailzed_box.y / image_shape[0]
        h = unnormailzed_box.h / image_shape[0]
        return BoundBox(x, y, w ,h)
    else:
        print("Before normalization:", unnormailzed_box)
        x = float(unnormailzed_box[0]) / image_shape[1]
        w = float(unnormailzed_box[2]) / image_shape[1]
        y = float(unnormailzed_box[1]) / image_shape[0]
        h = float(unnormailzed_box[3]) / image_shape[0]
        print("After normalization:", x, y, w, h)
        return [x, y, w, h]


def denormalize_box(image_shape, normailzed_box):
    """ DeNormalize box label from [0,1] to image size
    """
    if isinstance(normailzed_box, BoundBox):
        x = normailzed_box.x * image_shape[1]
        w = normailzed_box.w * image_shape[1]
        y = normailzed_box.y * image_shape[0]
        h = normailzed_box.h * image_shape[0]
    else:
        x = normailzed_box[0] * image_shape[1]
        w = normailzed_box[2] * image_shape[1]
        y = normailzed_box[1] * image_shape[0]
        h = normailzed_box[3] * image_shape[0]
    
    return [x, y, w ,h]

def parse_label(line):
    """ Parse (x_min, y_mix, w, h) label into (x_center, y_center, w, h)
    """
    x_min, y_min, w, h = line.split(',')
    x_min = float(x_min)
    y_min = float(y_min)
    w = float(w)
    h = float(h)

    x_center = (x_min + w/2)
    y_center = y_min + h/2

    box = BoundBox(x_center, y_center, w, h)

    return box
    
def make_box_label(box, image_shape):
    x = box.x * image_shape[1]
    w = box.w * image_shape[1]
    y = box.y * image_shape[0]
    h = box.h * image_shape[0]

    x_min = x - w/2
    y_min = y - h/2

    return [x_min, y_min, w, h]


def write_box_label(label_file, box, normalized, image_shape=None):
    """ Wirte box into label file
        input box: x_c, y_c, w, h
        output box: x_min, y_min, w, h
    """
    if normalized:
        x = box.x * image_shape[1]
        w = box.w * image_shape[1]
        y = box.y * image_shape[0]
        h = box.h * image_shape[0]
    else:
        x = box.x
        w = box.w
        y = box.y
        h = box.h

    x_min = x - w/2
    y_min = y - h/2
    label_file.write('{},{},{},{}\n'.format(int(x_min), int(y_min), int(w) ,int(h)))
    
        