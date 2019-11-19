import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt

file = '/home/smit/PycharmProjects/visDial_CPU/visDial.pytorch/script/data/vdl_img_vgg_demo.h5'
f = h5py.File(file)
imgs = f['images_train']

def old_vgg_16(idx):
    return torch.from_numpy(imgs[idx])