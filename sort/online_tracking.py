import os
import cv2
import glob
import math
import numpy as np
import matplotlib
import json
import time
from tqdm import tqdm
from os.path import basename, splitext

from sort import Sort
from bbox_utils import *
from sort_utils import sort_nicely
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from frontend import YOLO
from utils import BoundBox


def boxes2dets(boxes, image_shape):
    seq_dets = np.empty((len(boxes), 5))
    for i, box in enumerate(boxes):
        seq_dets[i, ...] = [box.x, box.y, box.w, box.h, box.c]

    # Scale [x,y,w,h] scale to image size and convert to [x1,y1,x2,y2]
    seq_dets[:, 0] *= image_shape[1]
    seq_dets[:, 1] *= image_shape[0]
    seq_dets[:, 2] *= image_shape[1]
    seq_dets[:, 3] *= image_shape[0]

    seq_dets[:, 0] -= seq_dets[:, 2] / 2
    seq_dets[:, 1] -= seq_dets[:, 3] / 2

    seq_dets[:, 2] += seq_dets[:, 0]
    seq_dets[:, 3] += seq_dets[:, 1]
    
    return seq_dets


def online_tracking(data_dir, STORE=False):
    if not os.path.exists(data_dir):
        raise IOError("Invalid data path:", data_dir)

    yolo_config_path = "../config_aerial.json"
    with open(yolo_config_path) as config_buffer:    
        yolo_config = json.load(config_buffer)

    yolo = YOLO(architecture        = yolo_config['model']['architecture'],
                input_size          = yolo_config['model']['input_size'], 
                labels              = yolo_config['model']['labels'], 
                max_box_per_image   = yolo_config['model']['max_box_per_image'],
                anchors             = yolo_config['model']['anchors'])
    yolo_weights_path = "../yolo_coco_aerial_person.h5"
    print("YOLO weights path:", yolo_weights_path)
    yolo.load_weights(yolo_weights_path)
    
    colours = np.round(np.random.rand(32, 3) * 255)

    frame_width = 1280
    frame_height = 720

    if STORE:
        video_name = data_dir.split('/')[-2]
        FPS = 30
        # remember to modify frame width and height before testing video
        video_writer = cv2.VideoWriter('output_video/' + video_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))

    image_paths = sorted(glob.glob(os.path.join(data_dir, '*jpg')))
    sort_nicely(image_paths)

    mot_tracker = Sort() # create instance of the SORT tracker

    total_time = 0.0

    for i, image_path in enumerate(tqdm(image_paths)):
        image = cv2.imread(image_path)
        
        track_start_time = time.time()

        boxes = yolo.predict(image)

        detect_time = time.time() - track_start_time
        
        sort_start_time = time.time()
        
        dets = boxes2dets(boxes, image.shape)
        trackers = mot_tracker.update(dets)

        end_time = time.time()
        cycle_time = end_time - track_start_time
        total_time += cycle_time
        sort_time = end_time - sort_start_time

        for d in trackers:
            color = colours[int(d[4])%32]
            cv2.rectangle(image, 
                         (int(d[0]), int(d[1])), 
                         (int(d[2]), int(d[3])),
                         color, 3)
            cv2.putText(image, 
                        'id = ' + str(int(d[4])), 
                        (int(d[0]), int(d[1]) - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image.shape[0], 
                        color, 2)

        cv2.putText(image, 
                    'Tracking FPS = {:.2f}'.format(1 / cycle_time),
                    (frame_width - 300, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,250,0), 2)

        cv2.putText(image, 
                    '   YOLO FPS = {:.2f}'.format(1 / detect_time),
                    (frame_width - 300, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,250,0), 2)

        cv2.putText(image, 
                    '   SORT FPS = {:.2f}'.format(1 / sort_time),
                    (frame_width - 300, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,250,0), 2)

        if STORE:
            video_writer.write(image)
        else:
            cv2.imshow("output", image)
            cv2.waitKey(0)

        if i > 450:
            break

    total_frames = i + 1
    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time, total_frames, total_frames/total_time))    


if __name__ == '__main__':
    # data_dir = '/home/peng/data/sort_data/images/person23/'
    data_dir = '/home/peng/data/UAV123/data_seq/UAV123/group1/'
    online_tracking(data_dir, STORE=True)
