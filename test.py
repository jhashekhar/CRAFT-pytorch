"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms.functional as TF

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_pytorch.craft_utils as craft_utils
import craft_pytorch.imgproc as imgproc
import craft_pytorch.file_utils as file_utils
import json
import zipfile

from craft_pytorch.craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
args = parser.parse_args()


""" For test images in a folder """

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

current_dir = os.path.join(os.getcwd(), 'uploads')
image_path = current_dir + '/' + os.listdir(current_dir)[0]
print(image_path)
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]


    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)


    return boxes, polys, ret_score_text


def labels_and_images(img_path, bboxes_list):
  # Transforms images into images and save them into a folder
  list_of_images = []
  count = 0
  for i in range(len(bboxes_list)):
    img = bboxes_list[i]

    x1, y1, x2, y2 = img[0], img[1], img[2], img[7]

    # Reading and saving image as tensors
    img = Image.open(img_path)
    x = TF.to_tensor(img)

    # :, second:last, first:third
    x_t = x[:, y1:y2, x1:x2]
    dest_dir = '/Users/xanthate/github/flask-tut/bboxes'
    torchvision.utils.save_image(x_t, '{}/bboxes_{}.jpg'.format(dest_dir, i))

    list_of_images.append('bboxes_{}.jpg'.format(i))
    count += 1
  return list_of_images


if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')

    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    net.eval()

    # load data
    print("Test image {:s}".format(image_path), end='\r')
    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)

    # save score text
    ###################filename, file_ext = os.path.splitext(os.path.basename(image_path))
    bboxes_list = []
    for box in bboxes:
        x = np.array(box).astype(np.int32).reshape((-1))
        x = x.tolist()
        bboxes_list.append(x)
    print("Length of bboxes_list: ", len(bboxes_list))
    loi = labels_and_images(image_path, bboxes_list)
    #print(bboxes_list, type(bboxes_list), type(bboxes_list[0]))


    #file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
