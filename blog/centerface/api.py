import os
import os.path as osp

import torch
import numpy as np
from PIL import Image

# local imports
from config import Config as cfg
from models.mnet import get_mobile_net
from utils import VisionKit


def load_model():
    net = get_mobile_net(10, {'hm':1, 'wh':2, 'lm':10, 'off':2}, head_conv=24)
    path = osp.join(cfg.checkpoints, cfg.restore_model)
    weights = torch.load(path, map_location=cfg.device)
    net.load_state_dict(weights)
    net.eval()
    return net

net = load_model()


def preprocess(im):
    new_im, _, _, *params = VisionKit.letterbox(im, cfg.insize)
    return new_im, params

def postprocess(bboxes, landmarks, params):
    bboxes, landmarks = VisionKit.letterbox_inverse(*params, bboxes, landmarks, skip=2)
    return bboxes, landmarks

def detect(im):
    data = cfg.test_transforms(im)
    data = data[None, ...]
    with torch.no_grad():
        out = net(data)
    return out[0]

def decode(out):
    hm = out['hm']
    wh = out['wh']
    off = out['off']
    lm = out['lm']
    hm = VisionKit.nms(hm, kernel=3)
    hm.squeeze_()
    off.squeeze_()
    wh.squeeze_()
    lm.squeeze_()
    
    hm = hm.numpy()
    hm[hm < cfg.threshold] = 0
    xs, ys = np.nonzero(hm)
    bboxes = []
    landmarks = []
    for x, y in zip(xs, ys):
        ow = off[0][x, y]
        oh = off[1][x, y]
        cx = (ow + y) * 4
        cy = (oh + x) * 4 

        w = wh[0][x, y]
        h = wh[1][x, y]
        width = np.exp(w) * 4
        height = np.exp(h) * 4
        
        left = cx - width / 2
        top = cy - height / 2
        right = cx + width / 2
        bottom = cy + height / 2
        bboxes.append([left, top, right, bottom])

        # landmark
        lms = []
        for i in range(0, 10, 2):
            lm_x = lm[i][x, y]
            lm_y = lm[i+1][x, y]
            lm_x = lm_x * width + left 
            lm_y = lm_y * height + top 
            lms += [lm_x, lm_y]
        landmarks.append(lms)
    return bboxes, landmarks


def visualize(im, bboxes, landmarks):
    return VisionKit.visualize(im, bboxes, landmarks, skip=2)


if __name__ == '__main__':
    impath = 'samples/d.jpg'
    im = Image.open(impath)
    new_im, params = preprocess(im)
    pred = detect(new_im)
    bboxes, landmarks = decode(pred)
    bboxes, landmarks = postprocess(bboxes, landmarks, params)
    visualize(im, bboxes, landmarks)