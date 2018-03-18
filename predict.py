#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import glob
import re


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='yolo_coco_person.h5',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


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

def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print("Weights path:", weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4' or image_path[-4:] == '.MOV':
        # video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_out = image_path[:-4] + '_detected.mp4'

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc('M','J','P','G'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  
    elif image_path[-4:] == '.jpg' or image_path[-4:] == '.png':
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print len(boxes), 'boxes are found'

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
    else:
        # image_path is a folder contains all the frames of a video
        image_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), image_path, '*jpg')))
        sort_nicely(image_paths)

        FPS = 50
        # remember to modify frame width and height before testing video
        frame_width = 960
        frame_height = 540
        video_writer = cv2.VideoWriter(image_path[:-1] + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))
        for i in tqdm(range(len(image_paths))):
            image = cv2.imread(image_paths[i])
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])
            video_writer.write(image)
            # cv2.imshow('Image', image)
            # cv2.waitKey(10)
        print('Video saved to:', image_path[:-1] + '.avi')

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
