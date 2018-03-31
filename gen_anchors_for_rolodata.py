import random
import argparse
import numpy as np
from kmeans import k_means
import math
from os.path import basename, splitext

from preprocessing import parse_annotation
from tracker.rolo_preprocessing import data_preparation
import json

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    default='config_aerial.json',
    help='path to configuration file')

argparser.add_argument(
    '-a',
    '--anchors',
    default=5,
    help='number of anchors to use')

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.2f,%0.2f, ' % (anchors[i,0], anchors[i,1])

    #there should not be comma after last anchor, that's why
    r += '%0.2f,%0.2f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"

    print r


def main(argv):
    config_path = args.conf
    num_anchors = args.anchors

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    data = {
        'train': {
            'image_folder': config['train']['train_image_folder'],
            'annot_folder': config['train']['train_annot_folder'],
        },
        'valid': {
            'image_folder': config['valid']['valid_image_folder'],
            'annot_folder': config['valid']['valid_annot_folder'],
        }
    }

    video_folder_list, video_annot_list = data_preparation(data['train'], FOR_YOLO=True)

    grid_w = config['model']['input_size']/32
    grid_h = config['model']['input_size']/32

    cell_w = 1280.0 / grid_w
    cell_h = 720.0 / grid_h

    # run k_mean to find the anchors
    annotation_dims = []
    for video_annot in video_annot_list:
        labels = np.loadtxt(video_annot, delimiter=',')
        for label in labels:
            relative_w = label[2] / cell_w
            relative_h = label[3] / cell_h
            if math.isnan(relative_w) or math.isnan(relative_h):
                # print("NaN annotations! {}".format(basename(video_annot)))
                1
            else:
                annotation_dims.append(map(float, (relative_w, relative_h)))
    annotation_dims = np.array(annotation_dims)

    centroids, cluster_assignment = k_means(annotation_dims, num_anchors)

    # write anchors to file
    print '\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids)
    print_anchors(centroids)

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
