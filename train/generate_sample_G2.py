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
from misc.keras_image_loader import vgg_16
from misc.old_vgg import old_vgg_16
import datetime
from misc.utils import repackage_hidden_new
from misc.Data_history import get_history_data, generate_ans_from_idx

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_h5', default='../script/data/vdl_img_vgg_demo.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_ques_h5', default='../script/data/visdial_data_demo.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='../script/data/visdial_params_demo.json', help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save', help='folder to output images and model checkpoints')
parser.add_argument('--encoder', default='G_QIH_VGG', help='what encoder to use.')
parser.add_argument('--model_path', default='/home/smit/saved_model/Generator-15/epoch_38.pth', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', default=20, help='number of image split out as validation set.')

parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start of epochs to train for')
parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
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
parser.add_argument('--gumble_weight', type=int, default=0.1, help='folder to output images and model checkpoints')
parser.add_argument('--log_interval', type=int, default=1, help='how many iterations show the log info')

opt = parser.parse_args()
print(opt)

print("=> loading checkpoint '{}'".format(opt.model_path))
checkpoint = torch.load(opt.model_path,map_location=torch.device('cpu'))
model_path = opt.model_path
opt.batchSize = 1

dataset_val = dl.validate(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'val')

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=5,
                                         shuffle=False, num_workers=int(opt.workers))

vocab_size = dataset_val.vocab_size # Current value 8964
ques_length = dataset_val.ques_length # 16
ans_length = dataset_val.ans_length + 1 # 9
his_length = dataset_val.ques_length + dataset_val.ans_length #24
itow = dataset_val.itow #index to word
img_feat_size = opt.conv_feat_size #512

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)

netW = model._netW(vocab_size, opt.ninp, opt.dropout)
netG = _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, opt.mos)
critG = model.LMCriterion()
sampler = model.gumbel_sampler()

netW.load_state_dict(checkpoint['netW'])
netE.load_state_dict(checkpoint['netE'])
netG.load_state_dict(checkpoint['netG'])

netE.eval()
netW.eval()
netG.eval()

img_input = torch.FloatTensor(opt.batchSize, 49, 512)
path_to_image = input("Insert path to image\n")
# path_to_image = '../images/dog1.jpg'
# returns tensor of pool5 of vgg16 for given image

img = vgg_16(path_to_image)
print(img.mean())
# idx = 9
# img = old_vgg_16(idx)
image = img.view(-1, img_feat_size)
img_input = torch.FloatTensor(opt.batchSize, 49, 512)
with torch.no_grad():
    img_input.resize_(image.size()).copy_(image)

caption = input("Write a caption\n")
questions = []
answers=[]
answers.append(caption)

while(True):
    question = input("Ask a question:\n")
    questions.append(question)
    ques, his = get_history_data(questions,answers)

    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    ind = his.size(1)

    his_input = torch.LongTensor(his.size())
    his_input.copy_(his)

    ques_input = torch.LongTensor(ques.size())
    ques_input.copy_(ques)

    ques_emb = netW(ques_input, format = 'index')
    his_emb = netW(his_input, format = 'index')

    ques_hidden = repackage_hidden_new(ques_hidden, 1)
    hist_hidden = repackage_hidden_new(hist_hidden, his_input.size(1))

    encoder_feat, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                        ques_hidden, hist_hidden, ind)

    _, ques_hidden = netG(encoder_feat.view(1,-1,opt.ninp), ques_hidden)

    # generate ans based on ques_hidden
    # using netG(x , ques_hidden)

    # Gumble softmax to sample the output.
    ans_length = 16
    fake_onehot = []
    fake_idx = []
    noise_input = torch.FloatTensor((ans_length, opt.batchSize, vocab_size + 1))
    noise_input.resize_(ans_length, opt.batchSize, vocab_size + 1)
    noise_input.data.uniform_(0, 1)

    ans_sample = torch.from_numpy(np.array([vocab_size]))
    # for di in range(ans_length):
    #     ans_emb = netW(ans_sample, format='index')
    #     logprob, ques_hidden = netG(ans_emb.view(1, -1, opt.ninp), ques_hidden)
    #     one_hot, idx = sampler(logprob, noise_input[di], opt.gumble_weight)
    #     fake_onehot.append(one_hot.view(1, -1, vocab_size + 1))
    #     fake_idx.append(idx)
    #     if di + 1 < ans_length:
    #         ans_sample = idx
    #
    # cur_ans = generate_ans_from_idx(fake_idx)
    # print('Answer: ',cur_ans)
    sample_ans_input = torch.LongTensor(1, opt.batchSize)
    sample_ans_input.resize_((1, opt.batchSize)).fill_(vocab_size)

    sample_opt = {'beam_size': 10}

    seq, seqLogprobs = netG.sample(netW, sample_ans_input, ques_hidden, sample_opt)
    ans_sample_txt = decode_txt(itow, seq.t())
    # print(type(ans_sample_txt[0]))
    print(ans_sample_txt[0])
    answers.append(ans_sample_txt[0])





