#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
from os.path import basename, splitext
import json
import glob
import re


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Detect videos and store in MOT format')

argparser.add_argument(
    '-c',
    '--conf',
    default='config_aerial.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='yolo_coco_aerial_person.h5',
    help='path to pretrained weights')


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

    ###############################
    #   Prepare data to be detected
    ###############################

    # data_folder = "/home/peng/data/good_rolo_data/"
    data_folder = "/home/peng/data/sort_data/images/"
    # data_folder = "/home/peng/data/sort_data/images/"
    video_folders_list = sorted(glob.glob(data_folder + '*'))
    sort_nicely(video_folders_list)
 
    config_path  = args.conf
    weights_path = args.weights

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

    for video_folder in video_folders_list:
        video_name = basename(video_folder)
        print("Processing %s." % video_name)
        image_paths = sorted(glob.glob(os.path.join(video_folder, '*jpg')))
        sort_nicely(image_paths)

        with open('det_mot/' + video_name + '.txt', 'w') as out_file:
            for i in tqdm(range(len(image_paths))):
                image = cv2.imread(image_paths[i])
                boxes = yolo.predict(image)

                for box in boxes:
                    x1 = (box.x - box.w/2) * 1280
                    y1 = (box.y - box.h/2) * 720
                    w = box.w * 1280
                    h = box.h * 720
                    print('%d,-1,%.2f,%.2f,%.2f,%.2f,%.6f,-1,-1,-1' % (i+1, x1, y1, w, h, box.c), file=out_file)
                    

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
