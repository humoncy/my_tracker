import tracker
import argparse
import json
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
        description='Recurrent YOLO')

argparser.add_argument(
    '-w',
    '--weights',
    default='checkpoints/rolo_overfitting-01-0.00.h5',
    help='path to rolo pretrained weights')

argparser.add_argument(
    '-c',
    '--config',
    default='rolo_config.json',
    help='path to rolo pretrained weights')


def _main_(args):

    yolo_config_path  = '../config.json'
    yolo_weights_path = '../yolo_coco_person.h5'
    rolo_config_path = args.config

    with open(yolo_config_path) as config_buffer:    
        yolo_config = json.load(config_buffer)

    with open(rolo_config_path) as config_buffer:
        rolo_config = json.load(config_buffer)

    # Modify the config properly before tracking!!

    batch_size = rolo_config["BATCH_SIZE"]
    time_step = rolo_config["TIME_STEP"]
    input_size = rolo_config["INPUT_SIZE"]
    cell_size = rolo_config["CELL_SIZE"]

    rolo = tracker.ROLO(
        batch_size = batch_size,
        time_step  = time_step,
        input_size = input_size,        
        cell_size  = cell_size,
        yolo_config = yolo_config,
        yolo_weights_path = yolo_weights_path
    )
    rolo.load_weights(rolo_config["rolo_pretrained_weight"])    
    rolo.track(rolo_config["test_video_folder"])    


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)