import os
import cv2
import glob
import re
import math
import numpy as np
from bbox_utils import xywh_to_xyxy

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    return l.sort(key=alphanum_key)

def check_data(data_dir, annot_path, YOLO_result=True):
    """
        input a video directory and its annotation file,
        and check if the annotations is correct or not
    """
    if not os.path.exists(data_dir):
        print("Invalid data path:", data_dir)
        exit()
    if not os.path.exists(annot_path):
        print("Invalid annotation path:", annot_path)
        exit()
    
    image_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), data_dir, '*jpg')))
    sort_nicely(image_paths)
    
    print(annot_path)

    if YOLO_result:
        bboxes = np.load(annot_path)
        bboxes[:, 0] *= 1280
        bboxes[:, 1] *= 720
        bboxes[:, 2] *= 1280
        bboxes[:, 3] *= 720
        bboxes[:, 0] -= bboxes[:, 2] / 2
        bboxes[:, 1] -= bboxes[:, 3] / 2
    else:
        bboxes = np.loadtxt(annot_path, delimiter=',')
    # print(bboxes.shape)
    bboxes = xywh_to_xyxy(bboxes)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        # print(image.shape)
        print("{}th frame".format(i))
        print(bboxes[i])
        if isNAN(bboxes[i]) is not True:
            cv2.rectangle(image, 
                (int(bboxes[i][0]),int(bboxes[i][1])), 
                (int(bboxes[i][2]),int(bboxes[i][3])), 
                (0,255,0), 3)
        cv2.imshow("output", image)
        cv2.waitKey(30)
        if i == 250:
            break

def isNAN(bbox):
    for value in bbox.flatten():
        if math.isnan(value):
            return True

def num_img(annot_folder_path):
    annot_file_list = sorted(glob.glob(os.path.join(os.path.dirname(__file__), annot_folder_path, '*txt')))
    sort_nicely(annot_file_list)

    num_total_imgs = 0
    for annot_file in annot_file_list:
        num_frames = sum(1 for line in open(annot_file))
        num_total_imgs += num_frames
        
    return num_total_imgs


if __name__ == '__main__':
    # data_dir = '/home/peng/data/rolo_data/images/train/person21/'
    # annot_path = '/home/peng/data/rolo_data/annotations/train/person21.txt'
    # check_data(data_dir, annot_path, YOLO_result=False)

    # data_dir = '/home/peng/data/rolo_data/images/train/person1/'
    # annot_path = '/home/peng/data/rolo_data/detected/train/person1.npy'
    # check_data(data_dir, annot_path ,YOLO_result=True)

    # data_dir = '/home/peng/data/rolo_data/images/valid/person14_2/'
    # annot_path = '/home/peng/data/rolo_data/detected/valid/person14_2npy'
    # check_data(data_dir, annot_path ,YOLO_result=True)

    num_train_img = num_img('/home/peng/data/rolo_data/annotations/train/')
    print(num_train_img)
