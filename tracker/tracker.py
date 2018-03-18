import argparse
import cv2
import numpy as np
from tqdm import tqdm
import json
import glob
import re

from bbox_utils import *
from rolo_preprocessing import BatchGenerator, data_preparation
from rolo_utils import sort_nicely

import os.path
from os.path import basename, splitext
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from frontend import YOLO
from utils import draw_boxes

from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers import CuDNNLSTM, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Recurrent YOLO')

argparser.add_argument(
    '-c',
    '--conf',
    default='../config.json',
    help='path to yolo configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='../yolo_coco_person.h5',
    help='path to yolo pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    default='/home/peng/data/aerial/person/000001.jpg',
    help='path to an an video folder')

argparser.add_argument(
    '-d',
    '--debug',
    action='store_true',
    default=False,
    help='Debug mode or not')

# image_path = '/home/peng/data/rolo_data/images/val/person/000001.jpg'


class ROLO(object):
    def __init__(self, 
        batch_size, 
        time_step, 
        input_size, 
        cell_size, 
        yolo_config, 
        yolo_weights_path):

        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size
        self.yolo_config = yolo_config
        self.yolo_weights_path = yolo_weights_path

        ##########################
        # Make the model
        ##########################

        inputs = Input(batch_shape=(self.batch_size, self.time_step, self.input_size))
        x = CuDNNLSTM(units=cell_size, return_sequences=True)(inputs)
        # Add output layer
        output = TimeDistributed(Dense(4))(x)

        self.model = Model(inputs, output)

    def load_weights(self, weight_path):
        print("Load pretrained weight:", weight_path)
        self.model.load_weights(weight_path)

    def detect_videos(self, data, mode):
        """ Detect videos by YOLO, andt store the detected bounding boxes and feature maps
        """
        
        video_folders_list, annotations_list = data_preparation(data[mode], FOR_YOLO=True)

        # ##############################
        #   Make the model 
        # ##############################

        yolo = YOLO(architecture        = self.yolo_config['model']['architecture'],
                    input_size          = self.yolo_config['model']['input_size'], 
                    labels              = self.yolo_config['model']['labels'], 
                    max_box_per_image   = self.yolo_config['model']['max_box_per_image'],
                    anchors             = self.yolo_config['model']['anchors'])

        # ###############################
        # #   Load trained weights
        # ###############################    

        print("YOLO weights path:", self.yolo_weights_path)
        yolo.load_weights(self.yolo_weights_path)

        for vid, video_folder in enumerate(video_folders_list):
            with open(annotations_list[vid], 'r') as annot_file:
                first_box_unnormailzed = parse_label(annot_file.readline())

            image_path_list = sorted(glob.glob(video_folder + "/*"))
            sort_nicely(image_path_list)

            first_image = cv2.imread(image_path_list[0])
            first_box = normalize_box(first_image.shape, first_box_unnormailzed)
            last_box = first_box

            # Write the detected labels into detected/
            detected_boxes = []
            detected_box = [first_box.x, first_box.y, first_box.w, first_box.h]
            detected_boxes.append(detected_box)

            detected_label_path = os.path.join(data[mode]['detected_folder'], basename(video_folder))

            # Write the detected features into features/
            features = []
            features_path = os.path.join(data[mode]['features_folder'], basename(video_folder))
            
            for i, image_path in enumerate(image_path_list):
                image = cv2.imread(image_path)
                if image is None:
                    print('Cannot find', image_path)
                boxes, feature = yolo.predict_for_rolo(image)
                chosen_box = choose_best_box(boxes, last_box)
                last_box = chosen_box                                

                # Write the detected images into detected_img/
                # detected_video_folder_path = os.path.join(data[mode]['detected_images_folder'], basename(video_folder))
                # if not os.path.exists(detected_video_folder_path):
                #     os.mkdir(detected_video_folder_path)
                # detected_img_path = os.path.join(detected_video_folder_path, basename(image_path))
                # cv2.imwrite(detected_img_path, image)

                if i > 0:
                    # Write the detected result of target
                    detected_box = [chosen_box.x, chosen_box.y, chosen_box.w, chosen_box.h]
                    detected_boxes.append(detected_box)

                # Write the detected features into features/
                features.append(feature)


                # Store YOLO detection result
                # image = draw_boxes(image, boxes, "person")
                # # print(len(boxes), 'boxes are found')
                # detected_video_folder_path = os.path.join(data[mode]['detected_images_folder'], self.yolo_config['model']['labels'])
                # if not os.path.exists(detected_video_folder_path):
                #     os.mkdir(detected_video_folder_path)
                # detected_img_path = os.path.join(detected_video_folder_path, basename(image_path))
                # cv2.imwrite(detected_img_path, image)
            
            

            print("======================= Save detected label result ==========================")
            detected_boxes = np.array(detected_boxes)
            print("Video:{} {} boxes are detected".format(basename(video_folder), detected_boxes.shape[0]))

            if DEBUG is not True:
                np.save(detected_label_path + '.npy', detected_boxes)
                # np.savetxt(detected_label_path + '.txt', detected_boxes, delimiter=',')
            else:
                print("-----Debugging-----")
                print("Write txt label file.")
                detected_boxes[:, 0] *= first_image.shape[1]
                detected_boxes[:, 1] *= first_image.shape[0]
                detected_boxes[:, 2] *= first_image.shape[1]
                detected_boxes[:, 3] *= first_image.shape[0]
                detected_boxes = np.round(detected_boxes)
                np.savetxt(detected_label_path + '.txt', detected_boxes.astype(int), fmt='%i', delimiter=',')

            print("========================== Save feature map =================================")
            features = np.array(features)
            np.save(features_path + '.npy', features)


    def train(self, 
              data_folder,
              train_times,
              valid_times,
              nb_epoch,
              learning_rate,
              saved_weights_name
            ):

        data = {
            'data_folder': data_folder,
            'train': {
                'image_folder': data_folder + 'images/train/',
                'annot_folder': data_folder + 'annotations/train/',
                'detected_folder': data_folder + 'detected/train/',
                'features_folder': data_folder + 'features/train/',
                'detected_images_folder': data_folder + 'detected_images/train/'
            },
            'valid': {
                'image_folder': data_folder + 'images/valid/',
                'annot_folder': data_folder + 'annotations/valid/',
                'detected_folder': data_folder + 'detected/valid/',
                'features_folder': data_folder + 'features/valid/',
                'detected_images_folder': data_folder + 'detected_images/valid/',
            }
        }

        ############################################
        # Detect your videos first if they have not been detected
        ############################################
        # self.detect_videos(data, 'train')
        # exit()
        # self.detect_videos(data, 'valid')

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'INPUT_SIZE'      : self.input_size, 
            'BATCH_SIZE'      : self.batch_size,
            'TIME_STEP'       : self.time_step
        }

        train_batch = BatchGenerator(data['train'],
                                     generator_config)

        valid_batch = BatchGenerator(data['valid'],
                                     generator_config)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
        file_path = "checkpoints/" + saved_weights_name[:-3] + "-{epoch:02d}-{val_loss:.4f}.h5"
        print("Save model path:", file_path)
        checkpoint = ModelCheckpoint(file_path, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        tb_counter  = len([log for log in os.listdir(os.path.expanduser('logs/')) if 'rolo' in log]) + 1
        tensorboard = TensorBoard(log_dir=os.path.expanduser('logs/') + 'rolo' + '_' + str(tb_counter), 
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################
        
        self.model.fit_generator(generator        = train_batch,
                                 steps_per_epoch  = len(train_batch) * train_times,   #TODO
                                 epochs           = nb_epoch,
                                 verbose          = 1,
                                 validation_data  = valid_batch,
                                 validation_steps = len(valid_batch) * valid_times,
                                 callbacks        = [early_stop, checkpoint, tensorboard],  #TODO
                                 workers          = 3
                                )
    

    def get_test_batch(self, inputs_list):
        test_batch = np.zeros((self.batch_size, self.time_step, self.input_size))
        for i in range(self.time_step):
            input_index = i + len(inputs_list) - self.time_step
            if input_index < 0:
                input_index = 0
            test_batch[:, i, ...] = inputs_list[input_index]
        
        return test_batch


    def track(self, video_folder_path):
        yolo = YOLO(architecture        = self.yolo_config['model']['architecture'],
                    input_size          = self.yolo_config['model']['input_size'], 
                    labels              = self.yolo_config['model']['labels'], 
                    max_box_per_image   = self.yolo_config['model']['max_box_per_image'],
                    anchors             = self.yolo_config['model']['anchors'])
        print("YOLO weights path:", self.yolo_weights_path)
        yolo.load_weights(self.yolo_weights_path)

        frame_path_list = sorted(glob.glob((video_folder_path + "*")))

        inputs_list = []

        initail_box = [810,165,50,111]  # x_min, y_min, w, h
        initail_box = xywh_xymin_to_xycenter(initail_box)  # x_center, y_center, w, h
        for i, frame_path in enumerate(frame_path_list):
            frame = cv2.imread(frame_path)
            if i == 0:
                frame = draw_box(frame, initail_box)
                boxes, feature = yolo.predict_for_rolo(frame)
                normalized_initial_box = normalize_box(frame.shape, initail_box)
                inputs = np.concatenate((feature.flatten(), normalized_initial_box))
                inputs_list.append(inputs)
                last_box = BoundBox(normalized_initial_box[0], normalized_initial_box[1], normalized_initial_box[2] ,normalized_initial_box[3])
            else:
                boxes, feature = yolo.predict_for_rolo(frame)
                chosen_box = choose_best_box(boxes, last_box)
                last_box = chosen_box
                chosen_box.print_box()
                inputs = np.concatenate((feature.flatten(), [chosen_box.x, chosen_box.y, chosen_box.w, chosen_box.h]))
                inputs_list.append(inputs)

                l_bound = i - self.time_step + 1
                if l_bound < 0:
                    l_bound = 0
                inputs = self.get_test_batch(inputs_list[l_bound:i+1])

                # Prediction by ROLO
                bbox = self.model.predict(inputs)[0, 1]
                frame = draw_box(frame, denormalize_box(frame.shape, chosen_box), color=(255,0,0))

                # Denormalize box
                bbox[0] *= frame.shape[1]
                bbox[1] *= frame.shape[0]
                bbox[2] *= frame.shape[1]
                bbox[3] *= frame.shape[0]
                frame = draw_box(frame, bbox)

            # cv2.imshow('video', frame)
            # cv2.waitKey(0)
            cv2.imwrite('output/' + str(i) + '.jpg', frame)


def _main_(args):
 
    yolo_config_path  = args.conf
    yolo_weights_path = args.weights
    rolo_config_path = "rolo_config.json"
    print("ROLO config file:", rolo_config_path)

    global DEBUG
    DEBUG = args.debug

    with open(yolo_config_path) as config_buffer:    
        yolo_config = json.load(config_buffer)

    ########################################################################################################
    # Modify the config properly before training!!
    with open(rolo_config_path) as config_buffer:
        config = json.load(config_buffer)
    ########################################################################################################

    batch_size = config["BATCH_SIZE"]
    time_step = config["TIME_STEP"]
    input_size = config["INPUT_SIZE"]
    cell_size = config["CELL_SIZE"]

    rolo = ROLO(
        batch_size = batch_size,
        time_step  = time_step,
        input_size = input_size,        
        cell_size  = cell_size,
        yolo_config = yolo_config,
        yolo_weights_path = yolo_weights_path
    )

    rolo.train(
        data_folder=config['data_folder'],
        train_times=10,
        valid_times=1,
        nb_epoch=5,
        learning_rate=1e-4,
        saved_weights_name='rolo_overfitting.h5'
    )

    # rolo.load_weights(config["rolo_pretrained_weight"])

    # rolo.track(config["test_video_folder"])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
