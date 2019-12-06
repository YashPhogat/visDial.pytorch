from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.insert(1,'/home/smit/PycharmProjects/visDial_CPU/visDial.pytorch/')

import pdb
import time
import numpy as np
import json
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
from misc.netG import _netG
from misc.image_loader import vgg_16
import datetime
from misc.utils import repackage_hidden_new
from misc.Data_history import get_history_data, generate_ans_from_idx

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_h5', default='../script/data/vdl_img_vgg_demo.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_ques_h5', default='../script/data/visdial_data_demo.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='../script/data/visdial_params_demo.json', help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save', help='folder to output images and model checkpoints')
parser.add_argument('--encoder', default='G_QIH_VGG', help='what encoder to use.')
parser.add_argument('--model_path', default='./save/Pre_trained_G/HCIAE-G-MLE.pth', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', default=20, help='number of image split out as validation set.')

parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start of epochs to train for')
parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--save_iter', type=int, default=2, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose'  , action='store_true', help='show the sampled caption')

parser.add_argument('--conv_feat_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--mos', action='store_true', help='whether to use Mixture of Softmaxes layer')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--gumble_weight', type=int, default=0.1, help='folder to output images and'
                                                                   ' model checkpoints')
parser.add_argument('--log_interval', type=int, default=1, help='how many iterations show the log info')

opt = parser.parse_args()

import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt

input_img_h5 = '../script/data/vdl_img_vgg_demo.h5'
path2 = '../script/data/visdial_params_demo.json'

f = json.load(open(path2, 'r'))
# self.itow = f['itow']
img_info = f['img_train']
print(img_info[0])

f = h5py.File(input_img_h5, 'r')
imgs = f['images_train'][0]
test1 = imgs

path_to_image = '../images/410561.jpg'
img = vgg_16(path_to_image)
print('a')
