from alphapose_module import estimate_pose
import os
import argparse
from skeleton_module import Skeleton
from evaluation_module import evaluate_exercise
import subprocess
import shlex
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Euclid Trainer')
    parser.add_argument('--cfg', type=str,
                        help='experiment configure file name', default="AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
    parser.add_argument('--checkpoint', type=str,
                        help='checkpoint file name', default="model/fast_res50_256x192.pth")
    parser.add_argument('--sp', default=True, action='store_true',
                        help='Use single process for pytorch')
    parser.add_argument('--detector', dest='detector',
                        help='detector name', default="yolo")
    parser.add_argument('--detfile', dest='detfile',
                        help='detection result file', default="")
    parser.add_argument('--indir', dest='inputpath',
                        help='image-directory', default="")
    parser.add_argument('--list', dest='inputlist',
                        help='image-list', default="")
    parser.add_argument('--image', dest='inputimg',
                        help='image-name', default="")
    parser.add_argument('--outdir', dest='outputpath',
                        help='output-directory', default="res/")
    parser.add_argument('--save_img', default=False, action='store_true',
                        help='save result as image')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualize image')
    parser.add_argument('--showbox', default=False, action='store_true',
                        help='visualize human bbox')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='add speed profiling at screen output')
    parser.add_argument('--format', type=str,
                        help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
    parser.add_argument('--min_box_area', type=int, default=0,
                        help='min box area to filter out')
    parser.add_argument('--detbatch', type=int, default=2,
                        help='detection batch size PER GPU')
    parser.add_argument('--posebatch', type=int, default=20,
                        help='pose estimation maximum batch size PER GPU')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                        help='save the result json as coco format, using image index(int) instead of image name(str)')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                        help='the length of result buffer, where reducing it will lower requirement of cpu memory')
    parser.add_argument('--flip', default=False, action='store_true',
                        help='enable flip testing')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print detail information')
    """----------------------------- Video options -----------------------------"""
    parser.add_argument('--video', dest='video',
                        help='video-name', default="")
    parser.add_argument('--webcam', dest='webcam', type=int,
                        help='webcam number', default=-1)
    parser.add_argument('--save_video', dest='save_video',
                        help='whether to save rendered video', default=False, action='store_true')
    parser.add_argument('--vis_fast', dest='vis_fast',
                        help='use fast rendering', action='store_true', default=False)
    """----------------------------- Tracking options -----------------------------"""
    parser.add_argument('--pose_flow', dest='pose_flow',
                        help='track humans in video with PoseFlow', action='store_true', default=False)
    parser.add_argument('--pose_track', dest='pose_track',
                        help='track humans in video with reid', action='store_true', default=False)
    """----------------------------- EuclidTrainer options -----------------------------"""         
    parser.add_argument("--exercise", dest='exercise',
                        help="type of exercise", type=str, default="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Make sure an exercise was specified
    assert args.exercise in ["squat", "curl"], "Please use --exercise to specify the exercise type.\nOptions are 'curl', 'squat'"

    # run AlphaPose on the video
    estimate_pose(args)

    # get height and width of the video in pixels
    file_path = args.video  
    vid = cv2.VideoCapture(file_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    # create the skeleton object
    skel = Skeleton("res/alphapose-results.json", V_HEIGHT=height, V_WIDTH=width)

    # evaluate the skeleton for the given lift
    evaluate_exercise(skel, exercise_type=args.exercise)