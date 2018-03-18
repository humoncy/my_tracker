import os
import cv2
import glob
import re
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

def check_data(data_dir, annot_path):
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
    bboxes = np.loadtxt(annot_path, delimiter=',', dtype=int)
    # print(bboxes.shape)
    bboxes = xywh_to_xyxy(bboxes)
    # print(bboxes.shape)
    

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        print(image.shape)
        print(bboxes[i])
        cv2.rectangle(image, 
            (bboxes[i][0],bboxes[i][1]), 
            (bboxes[i][2],bboxes[i][3]), 
            (0,255,0), 3)
        cv2.imshow("output", image)
        cv2.waitKey(30)


if __name__ == '__main__':
    data_dir = '/home/peng/data/rolo_data/images/train/person/'
    annot_path = '/home/peng/data/rolo_data/detected/train/person.txt'
    check_data(data_dir, annot_path)