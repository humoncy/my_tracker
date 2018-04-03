import tracker
import argparse
import json
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
        description='Recurrent YOLO')

argparser.add_argument(
    '-c',
    '--config',
    default='rolo_config.json',
    help='path to rolo pretrained weights')


# test_video_folder = "/home/peng/data/UAV123/data_seq/UAV123/person6/"

def _main_(args):

    rolo_config_path = args.config

    with open(rolo_config_path) as config_buffer:
        rolo_config = json.load(config_buffer)

    yolo_config_path  = rolo_config["yolo_config"]
    yolo_weights_path = rolo_config["yolo_weights"]
    
    with open(yolo_config_path) as config_buffer:    
        yolo_config = json.load(config_buffer)

    # Modify the config properly before tracking!!
    rolo = tracker.ROLO(
        mode = 'test',
        rolo_config = rolo_config,
        yolo_config = yolo_config,
        yolo_weights_path = yolo_weights_path
    )
    rolo.load_weights(rolo_config["test"]["weights"])

    if not os.path.exists(rolo_config["test"]["test_video_folder"]):
        raise IOError("Wrong image folder path:", rolo_config["test"]["test_video_folder"])
    else:
        print("Data folder:", rolo_config["test"]["test_video_folder"])
    if not os.path.exists(rolo_config["test"]["test_annot_file"]):
        raise IOError("Wrong annotation folder path:", rolo_config["test"]["test_annot_file"])
    else:
        print("Annotations folder:", rolo_config["test"]["test_annot_file"])

    test_video_folder = rolo_config["test"]["test_video_folder"]
    print("Video folder path:", test_video_folder)

    with open(rolo_config["test"]["test_annot_file"]) as annot_file:
        initial_box_str = annot_file.readline().strip().split(',')

    initial_box = [int(num_str) for num_str in initial_box_str]
    rolo.track(test_video_folder, initial_box)    


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
