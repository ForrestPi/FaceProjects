import os
import os.path as osp
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from .models import QualityNet
from .config import config as cfg


def load_model():
    cwd = osp.dirname(__file__)
    path = osp.join(cwd, cfg.checkpoints, cfg.restore_model)
    state_dict = torch.load(path, map_location=cfg.device)
    net = QualityNet()
    net = nn.DataParallel(net)
    net.to(cfg.device)
    net.load_state_dict(state_dict)
    net.eval()
    return net

net = load_model()

def input_preprocess(im):
    assert im.mode == 'RGB'
    im = cfg.test_transform(im)
    im = im[None, ...]
    return im

def _input_preprocess_cv(im):
    im = im[:, :, ::-1]
    im = Image.fromarray(im)
    return im
        
def estimate(im):
    global net
    if type(im) == str:
        im = Image.open(im)
    elif type(im) == np.ndarray: # opencv
        im = _input_preprocess_cv(im)
    im = input_preprocess(im)
    with torch.no_grad():
        score = net(im)
    score = round(score.item(), 2)
    return score