import argparse
import cv2
import numpy as np
from tqdm import tqdm
import json
import glob
import re
import time
import math

from bbox_utils import *
from rolo_preprocessing import BatchGenerator, data_preparation
from rolo_utils import sort_nicely, isNAN

import os.path
from os.path import basename, splitext
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from frontend import YOLO
# from utils import draw_boxes, bbox_iou, sigmoid, softmax

from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers import CuDNNLSTM, TimeDistributed, ConvLSTM2D, UpSampling2D, Add
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

def get_slice(x):
    return x[:, -1, 4]

class ROLO(object):
    def __init__(self,
        mode,
        rolo_config,
        yolo_config, 
        yolo_weights_path):

        self.batch_size = rolo_config[mode]['BATCH_SIZE']
        self.time_step = rolo_config[mode]['TIME_STEP']
        self.input_size = rolo_config['INPUT_SIZE']
        self.anchors = rolo_config["anchors"]
        self.yolo_config = yolo_config
        self.yolo_weights_path = yolo_weights_path

        # self.grid_w = 13
        # self.grid_h = 13
        # self.nb_class = 1
        # self.nb_box = 5
        # self.class_wt = np.ones(self.nb_class, dtype='float32')
        # self.true_boxes = Input(shape=(1, 1, 1, 1, 4))  

        ##########################
        # Make the model
        ##########################

        # Bad version
        # inputs = Input(batch_shape=(self.batch_size, self.time_step, self.input_size))
        # x = CuDNNLSTM(units=cell_size, return_sequences=True)(inputs)
        # # Add output layer
        # output = TimeDistributed(Dense(4))(x)

        if isinstance(self.input_size, list):
            inputs = Input(batch_shape=(self.batch_size, self.time_step, self.input_size[0], self.input_size[1], self.input_size[2]))
        else:
            inputs = Input(batch_shape=(self.batch_size, self.time_step, self.input_size))

        bbox_inputs = Input(batch_shape=(self.batch_size, self.time_step, 4))

        x = CuDNNLSTM(4, return_sequences=True, stateful=rolo_config[mode]['lstm_stateful'])(bbox_inputs)

        x = TimeDistributed(Dense(4))(x)
        x = Add()([x, bbox_inputs])
        output = TimeDistributed(Dense(4, activation='sigmoid'))(x)

        # self.model = Model(inputs, output)
        self.model = Model([inputs, bbox_inputs], output)

        # print a summary of the whole model
        self.model.summary()

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
            print(basename(video_folder))
            detected_label_path = os.path.join(data[mode]['detected_folder'], basename(video_folder))
            if os.path.exists(detected_label_path + '.npy') is True:
                continue

            features_path = os.path.join(data[mode]['features_folder'], basename(video_folder))
            if os.path.exists(features_path + '.npy') is True:
                continue
            
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
            features = []

            for i, image_path in enumerate(image_path_list):
                print("============ Detecting {} video, {} frame ===============".format(basename(video_folder), basename(image_path)))
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
            np.save(detected_label_path + '.npy', detected_boxes)

            print("========================== Save feature map =================================")
            features = np.array(features)
            np.save(features_path + '.npy', features)

    def coord_loss(self, y_true, y_pred):
        y_true *= 50
        y_pred *= 50

        # clip_min_value = tf.constant([1e-10, 1e-10])
        # min_tensor = tf.zeros((self.batch_size,2))
        # min_tensor = tf.add(min_tensor, clip_min_value)

        pred_box_xy = y_pred[..., :2]
        # pred_box_wh = tf.sqrt(tf.clip_by_value(y_pred[..., 2:], min_tensor, y_pred[..., 2:]))
        pred_box_wh = y_pred[..., 2:]

        true_box_xy = y_true[..., :2]
        # true_box_wh = tf.sqrt(tf.clip_by_value(y_true[..., 2:], min_tensor, y_true[..., 2:]))
        true_box_wh = y_true[..., 2:]

        loss_xy = tf.reduce_mean(tf.square(pred_box_xy - true_box_xy))
        loss_wh = tf.reduce_mean(tf.square(pred_box_wh - true_box_wh))

        loss = loss_xy + loss_wh

        if DEBUG:
            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)

        return loss

    def feat_loss(self, y_true, y_pred):
        y_true *= 50
        y_pred *= 50
        loss = tf.reduce_mean(tf.square(y_pred - y_true))

        return loss

        loss_score  = tf.reduce_sum(tf.square(y_true - y_pred))

        return loss_score

        mask_shape = tf.shape(y_true)[:4]
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, 5, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        tconf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust track confidence
        pred_box_tconf = tf.sigmoid(y_pred[..., 5])        
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 6:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]

        true_box_tconf = y_true[..., 5]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 6:], -1)
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale       

        # penalize the track confidence of the boxes, which are reponsible for corresponding ground truth box
        tconf_mask = tconf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 5]) * self.no_object_scale
        # penalize the track confidence of the boxes, which are reponsible for corresponding ground truth box
        tconf_mask = tconf_mask + y_true[..., 5] * self.object_scale

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_tconf  = tf.reduce_sum(tf.square(true_box_tconf-pred_box_tconf) * tconf_mask) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = loss_xy + loss_wh + loss_conf + loss_tconf + loss_class

        return loss

    def train(self, 
              data_folder,
              train_times,
              valid_times,
              nb_epoch,
              learning_rate,
              saved_weights_name
            ):

        self.object_scale    = self.yolo_config['train']['object_scale']
        self.no_object_scale = self.yolo_config['train']['no_object_scale']
        self.coord_scale     = self.yolo_config['train']['coord_scale']
        self.class_scale     = self.yolo_config['train']['class_scale']

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
        # self.model.compile(loss=self.custom_loss, optimizer=optimizer)
        self.model.compile(loss=self.coord_loss, optimizer=optimizer)
        # self.model.compile(loss={'feat_output': self.feat_loss, 'bbox_output': self.coord_loss}, loss_weights=[1., 1.], optimizer=optimizer)
        # self.model.compile(loss={'bbox_score_output': self.bbox_score_loss, 'output': self.custom_loss}, loss_weights=[1., 1.], optimizer=optimizer)
        # self.model.compile(loss='mse', optimizer=optimizer)

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'INPUT_SIZE'      : self.input_size, 
            'BATCH_SIZE'      : self.batch_size,
            'TIME_STEP'       : self.time_step,
            'GRID_H'          : 13,  
            'GRID_W'          : 13,
            'BOX'             : 5,
            'CLASS'           : 1,
            'ANCHORS'         : self.anchors
        }

        train_batch = BatchGenerator(data['train'],
                                     generator_config)

        valid_batch = BatchGenerator(data['valid'],
                                     generator_config)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.0001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
        file_path = "checkpoints/" + saved_weights_name[:-3] + "-{epoch:02d}-{val_loss:.2f}.h5"
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
                                 workers          = 1,   #TODO
                                 use_multiprocessing = False     #TODO
                                )
    

    def get_test_batch(self, inputs_list):
        """ Get test batch, if batch size is not 1, then duplicate the first batch to the others
        """
        if inputs_list is not None:
            input_shape = np.array(inputs_list[0]).shape[1:]
            test_batch_shape = [self.batch_size, self.time_step]
            test_batch_shape += list(input_shape)
            test_batch = np.zeros(tuple(test_batch_shape))
            
        for i in range(self.time_step):
            input_index = i + len(inputs_list) - self.time_step
            if input_index < 0:
                input_index = 0
            test_batch[:, i, ...] = inputs_list[input_index]
        
        return test_batch


    def track(self, video_folder_path, initial_box):
        yolo = YOLO(architecture        = self.yolo_config['model']['architecture'],
                    input_size          = self.yolo_config['model']['input_size'], 
                    labels              = self.yolo_config['model']['labels'], 
                    max_box_per_image   = self.yolo_config['model']['max_box_per_image'],
                    anchors             = self.yolo_config['model']['anchors'])
        print("YOLO weights path:", self.yolo_weights_path)
        yolo.load_weights(self.yolo_weights_path)

        frame_path_list = sorted(glob.glob((video_folder_path + "*")))
        if len(frame_path_list) == 0:
            raise IOError("Found {} frames".format(len(frame_path_list)))

        feature_inputs_list = []
        bbox_inputs_list = []

        initial_box = xywh_xymin_to_xycenter(initial_box)  # x_center, y_center, w, h

        tracking_time = 0.0

        for i, frame_path in enumerate(frame_path_list):
            print("================ {}th frame ==================".format(i))
            frame = cv2.imread(frame_path)
            if i == 0:
                frame = draw_box(frame, initial_box)
                boxes, feature = yolo.predict_for_rolo(frame)
                normalized_initial_box = normalize_box(frame.shape, initial_box)
                # inputs = np.concatenate((feature.flatten(), normalized_initial_box))
                # inputs = feature
                # inputs = normalized_initial_box
                feature_inputs_list.append(feature)
                bbox = np.expand_dims(np.array(normalized_initial_box), axis=0)
                bbox_inputs_list.append(bbox)
                
                last_box = BoundBox(normalized_initial_box[0], normalized_initial_box[1], normalized_initial_box[2] ,normalized_initial_box[3])
            else:
                boxes, feature = yolo.predict_for_rolo(frame)

                chosen_box = choose_best_box(boxes, last_box)
                last_box = chosen_box
                # chosen_box.print_box()
                # inputs = np.concatenate((feature.flatten(), [chosen_box.x, chosen_box.y, chosen_box.w, chosen_box.h]))
                # inputs = feature
                # inputs = [chosen_box.x, chosen_box.y, chosen_box.w, chosen_box.h]
                feature_inputs_list.append(feature)  # shape: [1,13,13,1024]
                bbox = np.expand_dims(np.array([chosen_box.x, chosen_box.y, chosen_box.w, chosen_box.h]), axis=0)  # shape: [1,4]
                bbox_inputs_list.append(bbox)

                l_bound = i - self.time_step + 1
                if l_bound < 0:
                    l_bound = 0
                feature_input = self.get_test_batch(feature_inputs_list[l_bound:i+1])
                bbox_input = self.get_test_batch(bbox_inputs_list[l_bound:i+1])

                # bbox_input = np.array([[chosen_box.x, chosen_box.y, chosen_box.w, chosen_box.h]])

                start_time = time.time()
                # Prediction by ROLO
                # bbox = self.model.predict([feature_input, bbox_input])[0, self.time_step - 1]
                predict_bbox = self.model.predict([feature_input, bbox_input])
                end_time = time.time()
                print("ROLO predict time: {} sec per image.".format(end_time-start_time))

                # end_time = time.time()

                # Draw detected box by YOLO
                detected_box = denormalize_box(frame.shape, chosen_box)
                frame = draw_box(frame, detected_box, color=(255,0,0))
                print("Detected box: [ {:.2f}  {:.2f}  {:.2f}  {:.2f} ]".format(detected_box[0], detected_box[1], detected_box[2], detected_box[3]))
                
                if i < self.time_step:
                    bbox = predict_bbox[0, i, ...]
                else:
                    bbox = predict_bbox[0, -1, ...]
                
                # Denormalize box
                bbox[0] *= frame.shape[1]
                bbox[1] *= frame.shape[0]
                bbox[2] *= frame.shape[1]
                bbox[3] *= frame.shape[0]
                # Draw Tracked box by ROLO
                frame = draw_box(frame, bbox)
                print("Tracked box : [ {:.2f}  {:.2f}  {:.2f}  {:.2f} ]".format(bbox[0], bbox[1], bbox[2], bbox[3]))

                tracking_time += (end_time - start_time)

            print("==============================================")
            

            # cv2.imshow('video', frame)
            # cv2.waitKey(0)
            cv2.imwrite('output/' + str(i) + '.jpg', frame)

            if i >= 100:
                break
        
        print("Tracking speed: {:.3f} FPS".format((len(frame_path_list) - 1) / tracking_time))


def _main_(args):

    global DEBUG
    DEBUG = args.debug
 
    rolo_config_path = "rolo_config.json"
    print("ROLO config file:", rolo_config_path)
    
    ########################################################################################################
    # Modify the config properly before training!!
    with open(rolo_config_path) as config_buffer:
        rolo_config = json.load(config_buffer)
    ########################################################################################################

    yolo_config_path  = rolo_config["yolo_config"]
    yolo_weights_path = rolo_config["yolo_weights"]

    with open(yolo_config_path) as config_buffer:    
        yolo_config = json.load(config_buffer)

    rolo = ROLO(
        mode = 'train',
        rolo_config = rolo_config,
        yolo_config = yolo_config,
        yolo_weights_path = yolo_weights_path
    )

    if rolo_config["train"]["use_pretrained_weight"] == "True":
        rolo.load_weights(rolo_config["train"]["rolo_pretrained_weight"])

    if rolo_config['warm_up'] == "True":
        data_folder = rolo_config['warm_up_data_folder']
        saved_weights_name = "warm_" + rolo_config["train"]["saved_weights_name"]
    else:
        data_folder = rolo_config['data_folder']
        saved_weights_name = rolo_config["train"]["saved_weights_name"]

    rolo.train(
        data_folder=data_folder,
        train_times=rolo_config["train"]["train_times"],
        valid_times=rolo_config["train"]["valid_times"],
        nb_epoch=rolo_config["train"]["nb_epoch"],
        learning_rate=rolo_config["train"]["learning_rate"],
        saved_weights_name=saved_weights_name
    )

    # rolo.track(config["test_video_folder"])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
