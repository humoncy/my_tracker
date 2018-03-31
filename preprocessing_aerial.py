import os
import cv2
import numpy as np
from os.path import basename

def aerial_parse_annotation(train_img_name_list, labels=[]):
    all_imgs = []
    seen_labels = {}
    

    with open(train_img_name_list) as f:
        image_path_list = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    image_path_list = [x.strip() for x in image_path_list]
    annot_path_list = [x.replace("Images", "Labels") for x in image_path_list]
    annot_path_list = [x.replace("jpg", "txt") for x in annot_path_list]

    for i, image_path in enumerate(image_path_list):
        img = {'object':[]}
        img['width'] = 1280
        img['height'] = 720
        img['filename'] = image_path
        with open(annot_path_list[i], 'r') as annot_file:
            annot = annot_file.readlines()
            annot = [x.strip() for x in annot]
            num_box =  int(annot[0])
            for j in range(num_box):
                obj = {}
                obj['name'] = 'person'
                bbox = annot[j+1].split(' ')
                obj['xmin'] = int(bbox[0])
                obj['ymin'] = int(bbox[1])
                obj['xmax'] = int(bbox[2])
                obj['ymax'] = int(bbox[3])
                img['object'] += [obj]
                
                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1

        all_imgs += [img]

                        
    return all_imgs, seen_labels