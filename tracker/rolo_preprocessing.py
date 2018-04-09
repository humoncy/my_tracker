import os
from os.path import basename, splitext
import cv2
import glob
import numpy as np
np.random.seed(41)
from rolo_utils import sort_nicely, isNAN
from keras.utils import Sequence


def data_preparation(data, FOR_YOLO=False):
    """ Return the paths to annotation files and image folders
        Arguments:
            data: config of the following form:
            {
                'image_folder': data_folder + 'images/train/',
                'annot_folder': data_folder + 'annotations/train/',
                'detected_folder': data_folder + 'detected/train/',
                'features_folder': data_folder + 'features/train/',
                'detected_images_folder': data_folder + 'detected_images/train/'
            }
            FOR_YOLO: need detected results or not
        Return:
            video_folders: list of video folder paths
            video_annotations: list of annotation file paths
    """
    print("\n========================== Preparing data ================================")
    print("\tConverting data into path name list.")
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

    video_folders = []
    detected_label_namelist = []
    features_namelist = []

    for i, annot_path in enumerate(video_annots):
        video_name = splitext(basename(annot_path))[0]   # Get the file name from its full path
        video_folder = os.path.join(data['image_folder'], video_name)
        if not os.path.exists(video_folder):
            raise IOError("Video folder does not exit:", video_folder)        
        video_folders.append(video_folder)

        if FOR_YOLO:
            continue

        detected_label_name = os.path.join(data['detected_folder'], video_name + '.npy')
        if not os.path.exists(detected_label_name):
            raise IOError("Detected label file does not exit:", detected_label_name)
        detected_label_namelist.append(detected_label_name)

        feature_name = os.path.join(data['features_folder'], video_name + '.npy')
        if not os.path.exists(feature_name):
            raise IOError("Feature map file does not exit:", feature_name)
        features_namelist.append(feature_name)


    print("Video folders:", video_folders)
    print("Annotations files:", video_annots)
    print("========================== Data prepared =================================\n")

    if FOR_YOLO:
        return video_folders, video_annots
    else:
        return video_folders, video_annots, detected_label_namelist, features_namelist


def random_select(annotations, time_step, batch_size):
    selected_video_num = np.random.random_integers(0, len(annotations)-1)

    num_frames = sum(1 for line in open(annotations[selected_video_num]))
    selected_frame_l_bound = np.random.random_integers(0, len(num_frames)-1)
    selected_frame_r_bound = selected_frame_l_bound + time_step
    if selected_frame_r_bound > num_frames:
        selected_frame_r_bound = num_frames
        selected_frame_l_bound = selected_frame_r_bound - time_step

    return selected_video_num, selected_frame_l_bound, selected_frame_r_bound


class BatchGenerator(Sequence):
    def __init__(self, data, config, shuffle=False):
        """
        Arguments:
            data: a dictionary contains the data folder information
            data['image_folder']: the folder containing training or validation images
                ``` Folder hierarchy:
                    - data_folder/
                        - video_1/
                            - frame 0
                            - frame 1
                            - ...
                        - video_2/
                            - frame 0
                            - frame 1
                            - ...
                        ....
                ```
            data['annot_folder']: the folder containing training or validation annotations
                ``` Folder hierarchy:
                    - annotations_folder/
                        - video_1.txt
                        - video_2.txt
                        - ...
                ```
            config: a dictionary stores the information of the model
                ``` Config example:
                    generator_config = {
                        'IMAGE_H'         : self.input_size, 
                        'IMAGE_W'         : self.input_size,
                        'BATCH_SIZE'      : self.batch_size,
                        'TIME_STEP'       : self.time_step
                    }
                ```
        """

        self.generator = None
        self.config = config
        self.shuffle = shuffle

        video_folders, annotations, detected_label_namelist, features_namelist = data_preparation(data)

        self.video_folders = video_folders   # name list of video folders
        self.annotations = annotations       # name list of video annotations
        self.detected_label_namelist = detected_label_namelist
        self.features_namelist = features_namelist

        self.num_images = len(annotations) * 500  # Guess the total number of images in the dataset #TODO

    def __len__(self):
        return int(np.ceil(float(self.num_images) / self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        # 1. Random select a video
        # 2. slice the video frames into batches and select one batch([l_bound, r_bound])
        selected_video_num = np.random.random_integers(0, len(self.annotations)-1)
        num_frames = sum(1 for line in open(self.annotations[selected_video_num]))
        num_batch = num_frames / self.config['BATCH_SIZE']
        l_bound = (idx % num_batch) * self.config['BATCH_SIZE']
        r_bound = (idx % num_batch + 1) * self.config['BATCH_SIZE']
        if r_bound - 1 + self.config['TIME_STEP'] > num_frames:
            r_bound = num_frames - self.config['TIME_STEP'] + 1
            l_bound = r_bound - self.config['BATCH_SIZE']
            if l_bound < 0:
                raise Exception("Number of frames in every video must be more than batch size ( > %d )" % self.config['BATCH_SIZE'])
        if (l_bound + self.config['BATCH_SIZE']-1 + self.config['TIME_STEP']-1) > num_frames - 1:
            l_bound = num_frames-1 - (self.config['BATCH_SIZE']-1 + self.config['TIME_STEP']-1)
            r_bound = l_bound + self.config['BATCH_SIZE']
        
        labels = np.loadtxt(self.annotations[selected_video_num], delimiter=',')

        ##########################################################################
        # Make sure labels are not NAN
        ##########################################################################
        
        while isNAN(labels[l_bound:(r_bound - 1 + self.config['TIME_STEP']), ...]):
            # print("\nGround truth is Nan, choose another batch")
            l_bound = (np.random.random_integers(0, 1000) % num_batch) * self.config['BATCH_SIZE']
            r_bound = l_bound + self.config['BATCH_SIZE']
            if r_bound - 1 + self.config['TIME_STEP'] > num_frames:
                r_bound = num_frames - self.config['TIME_STEP'] + 1
                l_bound = r_bound - self.config['BATCH_SIZE']
                if l_bound < 0:
                    raise Exception("Number of frames in every video must be more than batch size ( > %d )" % self.config['BATCH_SIZE'])
            # print("l_bound:", l_bound)
            # print("r_bound:", r_bound)
            # print("#frame:", num_frames)
            # print('#batch:', num_batch)
        

        detections = np.load(self.detected_label_namelist[selected_video_num])
        features = np.load(self.features_namelist[selected_video_num])

        # TODO
        labels[:, 0] = (labels[:, 0] + labels[:, 2] / 2.0) / 1280
        labels[:, 1] = (labels[:, 1] + labels[:, 3] / 2.0) / 720
        labels[:, 2] = labels[:, 2] / 1280
        labels[:, 3] = labels[:, 3] / 720

        # Make input data
        if isinstance(self.config['INPUT_SIZE'], list):
            x_batch = np.zeros((self.config['BATCH_SIZE'], self.config['TIME_STEP'], 
                                self.config['INPUT_SIZE'][0], self.config['INPUT_SIZE'][1], self.config['INPUT_SIZE'][2]))
            bbox_batch = np.zeros((self.config['BATCH_SIZE'], self.config['TIME_STEP'], 4))
            y_batch = np.zeros((self.config['BATCH_SIZE'], 4))
            feat_batch = np.zeros((self.config['BATCH_SIZE'], self.config['INPUT_SIZE'][0], self.config['INPUT_SIZE'][1], self.config['INPUT_SIZE'][2]))
        else:
            x_batch = np.zeros((self.config['BATCH_SIZE'], self.config['TIME_STEP'], self.config['INPUT_SIZE']))
            y_batch = np.zeros((self.config['BATCH_SIZE'], self.config['TIME_STEP'], 4))

        instance_count = 0        
        for i in range(l_bound,r_bound):
            # Every instance in every batch contains #time_step images
            for j in range(self.config['TIME_STEP']):
                detection = detections[i+j, ...]
                feature = features[i+j, ...]
                # label = labels[i+j, ...]
                # print("Detection shape:", detection.shape)
                # print("Feature shape:", feature.shape)
                # print("Detection:", detection)
                # inputs = detection
                # inputs = np.concatenate((feature.flatten(), detection))
                # inputs = feature
                # print("Input shape:", inputs.shape)

                x_batch[instance_count, j, :] = feature

                if isNAN(detection):
                    # When no detection results in some frame, use the initial box plus some Gaussian random normals as detection results
                    print("No detection results, use auxiliary bbox.")
                    detection = detections[0, ...]
                    for value in detection:
                        tmp = value + np.random_normal(0, 0.5)
                        while tmp < 0 or tmp > 1:
                            tmp = value + np.random_normal(0, 0.5)
                        value = tmp
                bbox_batch[instance_count, j, :] = detection

                # y_batch[instance_count, j, :] = label
                if j == self.config['TIME_STEP'] - 1:
                    # detection = detections[i+j, ...]
                    # if isNAN(detection):
                    #     detection = detections[0, ...]
                    #     for value in detection:
                    #         tmp = value + np.random_normal(0, 0.5)
                    #         while tmp < 0 or tmp > 1:
                    #             tmp = value + np.random_normal(0, 0.5)
                    #         value = tmp
                        # print(detection)
                        
                    # bbox_batch[instance_count, :] = detection
                    label = labels[i+j, ...]
                    if isNAN(label):
                        raise ValueError("Label is nan!")
                    for value in label:
                        if value < 0:
                            print(label)
                            raise ValueError("Label value should > 0")
                    # print(label)
                    y_batch[instance_count, :] = label
                    feat_batch[instance_count, ...] = feature
                # print("Label:", label)

            instance_count += 1

        if isNAN(x_batch):
            raise ValueError("X batch NAN!!!!!!!!!!!!!1")
        
        if isNAN(bbox_batch):
            raise ValueError("BBOX batch NAN!!!!!!!!!!!!!1")

        # print ' new batch created', idx
        # print(feat_batch.shape)
        return [x_batch, bbox_batch], [feat_batch, y_batch]
    
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
