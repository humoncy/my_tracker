import os
import cv2
import glob
import re
import math
import numpy as np
from bbox_utils import xywh_to_xyxy
from os.path import basename, splitext


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

def check_data(data_dir, annot_path, det_path, sort_result_path=None, format=None, STORE=False):
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

    if STORE is True:
        video_name = basename(annot_path)[:-4]
        FPS = 30
        # remember to modify frame width and height before testing video
        frame_width = 1280
        frame_height = 720
        video_writer = cv2.VideoWriter(video_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))
    
    image_paths = sorted(glob.glob(os.path.join(data_dir, '*jpg')))
    sort_nicely(image_paths)
    
    if format == 'YOLO':
        bboxes = np.load(det_path)
        bboxes[:, 0] *= 1280
        bboxes[:, 1] *= 720
        bboxes[:, 2] *= 1280
        bboxes[:, 3] *= 720
        bboxes[:, 0] -= bboxes[:, 2] / 2
        bboxes[:, 1] -= bboxes[:, 3] / 2
    elif format == 'MOT_FORMAT':
        mot_format = np.loadtxt(det_path, delimiter=',')
        sort_result = np.loadtxt(sort_result_path, delimiter=',')
    elif format == 'SORT_RESULT':
        det_boxes = np.load(det_path)
        det_boxes[:, 0] *= 1280
        det_boxes[:, 1] *= 720
        det_boxes[:, 2] *= 1280
        det_boxes[:, 3] *= 720
        det_boxes[:, 0] -= det_boxes[:, 2] / 2
        det_boxes[:, 1] -= det_boxes[:, 3] / 2

        mot_format = np.loadtxt(sort_result_path, delimiter=',')
        bboxes = mot_format[:, 2:6]
    else:
        bboxes = np.loadtxt(det_path, delimiter=',')

    annot_bboxes = np.loadtxt(annot_path, delimiter=',')
    # [x1 y1 w h] to [x1 y1 x2 y2]
    annot_bboxes = xywh_to_xyxy(annot_bboxes)

    if format == 'SORT_RESULT':
        det_boxes = xywh_to_xyxy(det_boxes)
        mot_format[:, 2:6] = xywh_to_xyxy(mot_format[:, 2:6])

    if format == 'MOT_FORMAT':
        mot_format[:, 2:6] = xywh_to_xyxy(mot_format[:, 2:6])
        sort_result[:, 2:6] = xywh_to_xyxy(sort_result[:, 2:6])
        
    else:
        bboxes = xywh_to_xyxy(bboxes)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        # print(image.shape)
        print("{}th frame".format(i))
        # print(bboxes[i])

        if format == 'SORT_RESULT':
            # Track result
            index_list = np.argwhere(mot_format[:, 0] == (i+1))
            if index_list.shape[0] != 0:
                index = index_list[0, 0]
                cv2.rectangle(image, 
                    (int(mot_format[index, 2]),int(mot_format[index, 3])), 
                    (int(mot_format[index, 4]),int(mot_format[index, 5])), 
                    (0,255,0), 3)
                print(mot_format[index, 2:6])
            # Detection result
            cv2.rectangle(image, 
                (int(det_boxes[i][0]),int(det_boxes[i][1])), 
                (int(det_boxes[i][2]),int(det_boxes[i][3])), 
                (255,0,0), 3)
            print(det_boxes[i, 0:4])
        elif format == 'MOT_FORMAT':
            # Detect result
            index_list = np.argwhere(mot_format[:, 0] == (i+1))
            if index_list.shape[0] != 0:
                for index in index_list[:, 0]:
                    if mot_format[index, 6] > 0.4:
                        cv2.rectangle(image, 
                            (int(mot_format[index, 2]),int(mot_format[index, 3])), 
                            (int(mot_format[index, 4]),int(mot_format[index, 5])), 
                            (255,0,0), 3)
                        print(mot_format[index, 2:6])
            # Track result
            index_list = np.argwhere(sort_result[:, 0] == (i+1))
            if index_list.shape[0] != 0:
                for index in index_list[:, 0]:
                    cv2.rectangle(image, 
                        (int(sort_result[index, 2]),int(sort_result[index, 3])), 
                        (int(sort_result[index, 4]),int(sort_result[index, 5])), 
                        (0,255,0), 3)
                    print(sort_result[index, 2:6])
        else:
            cv2.rectangle(image, 
                (int(bboxes[i][0]),int(bboxes[i][1])), 
                (int(bboxes[i][2]),int(bboxes[i][3])), 
                (255,0,0), 3)

        if isNAN(annot_bboxes[i]) is not True:
            cv2.rectangle(image, 
                (int(annot_bboxes[i][0]),int(annot_bboxes[i][1])), 
                (int(annot_bboxes[i][2]),int(annot_bboxes[i][3])), 
                (0,0,255), 3)

        if STORE:
            video_writer.write(image)
        else:
            cv2.imshow("output", image)
            try:
                if index_list.shape[0] == 0:
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(30)
            except:
                cv2.waitKey(30)
            
        # if i > 20 or i + 1 > len(mot_format)-1:
        #     break

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
    # data_dir = '/home/peng/data/UAV123/data_seq/UAV123/uav7/'
    # annot_path = '/home/peng/data/UAV123/anno/UAV123/uav7.txt'
    # check_data(data_dir, annot_path)

    # data_dir = '/home/peng/data/rolo_data/images/train/person1/'
    # annot_path = '/home/peng/data/rolo_data/detected/train/person1.npy'
    # check_data(data_dir, annot_path ,format='YOLO')

    data_dir   = '/home/peng/data/sort_data/images/person4_1/'
    annot_path = '/home/peng/data/sort_data/annotations/person4_1.txt'
    det_path   = '/home/peng/basic-yolo-keras/det_mot/person4_1.txt'
    sort_result= '/home/peng/basic-yolo-keras/sort/output/person4_1.txt'
    check_data(data_dir, annot_path, det_path, sort_result_path=sort_result, format='MOT_FORMAT')

    # data_dir   = '/home/peng/data/sort_data/images/person22/'
    # annot_path = '/home/peng/data/sort_data/annotations/person22.txt'
    # det_path = '/home/peng/data/sort_data/detected/person22.npy'
    # check_data(data_dir, annot_path, det_path, format='YOLO', STORE=True)

    # data_dir   = '/home/peng/data/sort_data/images/person14_3/'
    # annot_path = '/home/peng/data/sort_data/annotations/person14_3.txt'
    # det_path = '/home/peng/data/sort_data/detected/person14_3.npy'
    # sort_result= '/home/peng/basic-yolo-keras/sort/output/person14_3.txt'
    # check_data(data_dir, annot_path, det_path, sort_result_path=sort_result, format='SORT_RESULT', STORE=False)

    # num_train_img = num_img('/home/peng/data/rolo_data/annotations/train/')
    # print(num_train_img)
