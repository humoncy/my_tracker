import glob
import sys
import os.path
import cv2
import re
import argparse

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

argparser = argparse.ArgumentParser(
    description='Convert video frames to video')

argparser.add_argument(
    '-d',
    default='output/',
    metavar='DIR',
    dest='dir',
    help='path to the directory of video frames')

argparser.add_argument(
    '-o',
    default='output',
    metavar='OUTPUT',
    dest='output',
    help='output video name, default save to .avi format')

def _main_(args):
    video_frames_path = args.dir
    output_name = args.output
    if output_name[-4:] == ".avi":
        1
    else:
        output_name = args.output + ".avi"

    print("Video path:", os.path.abspath(os.path.join(os.path.dirname(__file__), video_frames_path)))
    if not os.path.exists(video_frames_path):
        print("Invalid video path, please modify the input or the script for a bit.")
        exit(0)

    image_paths = sorted(glob.glob(video_frames_path +'/*.jpg'))
    sort_nicely(image_paths)

    img = cv2.imread(image_paths[0])
    frame_width = img.shape[1]
    frame_height = img.shape[0]
    FPS = 30

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))

    for image_path in image_paths:
        image = cv2.imread(image_path)
        out.write(image)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
