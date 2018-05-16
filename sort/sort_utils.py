import os
import cv2
import glob
import re
import math
import numpy as np
from os.path import basename, splitext
import json

from bbox_utils import *
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from frontend import YOLO


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


def get_data_lists(data, MOT=False):
    """ Prepare rolo data for SORT
        Arguments:
            data: config of the following form:
            {
                'image_folder': data_folder + 'images/train/',
                'annot_folder': data_folder + 'annotations/train/',
                'detected_folder': data_folder + 'detected/train/',
                'sort_det_folder': data_folder + 'sort/train/'
            }
        Returns:
            video_folders: list of video folder paths
            video_annotations: list of annotation file paths
            det_list  : path name list of detected results
    """

    if not os.path.exists(data['image_folder']):
        raise IOError("Wrong image folder path:", data['image_folder'])
    else:
        print("Data folder:", data['image_folder'])
    if not os.path.exists(data['annot_folder']):
        raise IOError("Wrong annotation folder path:", data['annot_folder'])
    else:
        print("Annotations folder:", data['annot_folder'])

    # Get the annotations as a list: [video1ann.txt, video2ann.txt, video3ann.txt, ...]
    video_annots = sorted(glob.glob((data['annot_folder'] + "*")))
    sort_nicely(video_annots)

    if not os.path.exists(data['detected_folder']):
        os.makedirs(data['detected_folder'])
    else:
        print("Detected folder:", data['detected_folder'])

    if len(glob.glob((data['detected_folder'] + "*"))) < len(video_annots):
        print(len(glob.glob((data['detected_folder'] + "*"))))
        print(len(video_annots))
        exit()
        video_folders_list = sorted(glob.glob((data['image_folder'] + '*')))
        sort_nicely(video_folders_list)
        detect_videos(video_annots, video_folders_list, data['detected_folder'])

    video_folders = []
    det_list = []

    for i, annot_path in enumerate(video_annots):
        video_name = splitext(basename(annot_path))[0]   # Get the file name from its full path
        video_folder = os.path.join(data['image_folder'], video_name)
        if not os.path.exists(video_folder):
            raise IOError("Video folder does not exit:", video_folder)        
        video_folders.append(video_folder)

        detected_name = os.path.join(data['detected_folder'], video_name + '.npy')
        if not os.path.exists(detected_name):
            raise IOError("Detected file does not exit:", detected_name)

        if not os.path.exists(data['sort_det_folder']):
            os.makedirs(data['sort_det_folder'])

        if MOT:
            mot_det_path = data['sort_det_folder'] + video_name + '.txt'
        else:
            mot_det_path = change_box_format(detected_name, data['sort_det_folder'], video_name)
        det_list.append(mot_det_path)

    return video_annots, video_folders, det_list

def change_box_format(det_file, sort_det_folder, video_name):
    ''' Detection results from YOLO are normalized bbox [x_center, y_center, w, h]
        Change to MOT format first
    '''
    detections = np.load(det_file)
    detections[:, 0] *= 1280
    detections[:, 1] *= 720
    detections[:, 2] *= 1280
    detections[:, 3] *= 720
    detections[:, 0] -= detections[:, 2] / 2
    detections[:, 1] -= detections[:, 3] / 2

    output_filepath = sort_det_folder + video_name + '.txt'
    # print(output_filepath)

    nb_frame = detections.shape[0]

    mot_format = np.ones((nb_frame, 10))
    mot_format *= -1
    mot_format[:, 0] = np.arange(1, nb_frame+1)
    mot_format[:, 2:6] = detections
    mot_format[:, 6] = 0.8

    save_format = ['%i', '%i', '%.2f', '%.2f', '%.3f', '%.3f', '%.6f', '%i', '%i', '%i']
    np.savetxt(output_filepath, mot_format, delimiter=',', fmt=save_format)

    return output_filepath

def detect_videos(annotations_list, video_folders_list, detected_folder):
    """ Detect videos by YOLO, and store the detected bounding boxes
    """
    yolo_config_path = "../config_aerial.json"
    with open(yolo_config_path) as config_buffer:    
        yolo_config = json.load(config_buffer)

    # ##############################
    #   Make the model 
    # ##############################

    yolo = YOLO(architecture        = yolo_config['model']['architecture'],
                input_size          = yolo_config['model']['input_size'], 
                labels              = yolo_config['model']['labels'], 
                max_box_per_image   = yolo_config['model']['max_box_per_image'],
                anchors             = yolo_config['model']['anchors'])

    # ###############################
    # #   Load trained weights
    # ###############################    

    yolo_weights_path = "../yolo_coco_aerial_person.h5"
    print("YOLO weights path:", yolo_weights_path)
    yolo.load_weights(yolo_weights_path)

    if len(annotations_list) != len(video_folders_list):
        raise IOError("Mismatch # videos {} {}.".format(len(annotations_list), len(video_folders_list)))

    for vid, video_folder in enumerate(video_folders_list):
        print(basename(video_folder))
        detected_label_path = os.path.join(detected_folder, basename(video_folder))
        if os.path.exists(detected_label_path + '.npy'):
            continue

        if basename(annotations_list[vid]) != (basename(video_folder) + ".txt"):
            print("Annot: {}".format(basename(annotations_list[vid])))
            print("image: {}".format(basename(video_folder)))
            raise IOError("Mismatch video {}.".format(basename(video_folder)))

        num_frames = sum(1 for line in open(annotations_list[vid], 'r'))
        image_path_list = sorted(glob.glob(video_folder + "/*"))
        sort_nicely(image_path_list)

        if num_frames != len(image_path_list):
            raise IOError("Number of frames in {} does not match annotations.".format(basename(video_folder)))

        with open(annotations_list[vid], 'r') as annot_file:
            first_box_unnormailzed = parse_label(annot_file.readline())

        first_image = cv2.imread(image_path_list[0])
        first_box = normalize_box(first_image.shape, first_box_unnormailzed)
        last_box = first_box

        # Write the detected labels into detected/
        detected_boxes = []
        detected_box = [first_box.x, first_box.y, first_box.w, first_box.h]
        detected_boxes.append(detected_box)

        # Write the detected features into features/

        for i, image_path in enumerate(image_path_list):
            print("============ Detecting {} video, {} frame ===============".format(basename(video_folder), basename(image_path)))
            image = cv2.imread(image_path)
            if image is None:
                print('Cannot find', image_path)
            boxes, dummy_feature = yolo.predict_for_rolo(image)
            chosen_box = choose_best_box(boxes, last_box)
            last_box = chosen_box                                

            if i > 0:
                # Write the detected result of target
                detected_box = [chosen_box.x, chosen_box.y, chosen_box.w, chosen_box.h]
                detected_boxes.append(detected_box)

        print("======================= Save detected label result ==========================")
        detected_boxes = np.array(detected_boxes)
        print("Video:{} {} boxes are detected".format(basename(video_folder), detected_boxes.shape[0]))
        np.save(detected_label_path + '.npy', detected_boxes)
