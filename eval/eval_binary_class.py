from __future__ import print_function
import argparse
import os
import random
import sys

sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json

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
import h5py

# from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
#                     decode_txt, sample_batch_neg, l2_norm
# import misc.dataLoader as dl
# import misc.model as model
# from misc.encoder_QIH import _netE
# import datetime
# from misc.netG import _netG

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_h5', default='../script/data/vdl_img_vgg.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_ques_h5', default='../script/data/visdial_data.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='../script/data/visdial_params.json', help='path to dataset, now hdf5 file')

parser.add_argument('--model_path', default='/home/smit/saved_model/Generator-15/epoch_30.pth', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--path_to_home', type=str)
parser.add_argument('--evalall', action='store_true')
parser.add_argument('--early_stop', type=int, default='1000000', help='datapoints to consider')
parser.add_argument('--file_name', type=str)
parser.add_argument('--debug', action='store_true', help='enables cuda')
opt = parser.parse_args()

sys.path.insert(1, '../')

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
    decode_txt, sample_batch_neg, l2_norm
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_QIH import _netE
from misc.netG import _netG
import datetime
from misc.utils import repackage_hidden_new
from misc.Data_history import get_history_data, generate_ans_from_idx

# json output path
pth = os.path.split(opt.model_path)
tail = pth[1]
tail2 = tail[:-4]
json_path = tail2 + '_top10.json'
print('output will be dumped to: ' + json_path)

opt.manualSeed = random.randint(1, 10000)  # fix seed
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print("=> loading checkpoint '{}'".format(opt.model_path))
checkpoint = torch.load(opt.model_path,map_location=torch.device('cpu'))
model_path = opt.model_path
# data_dir = opt.data_dir
input_img_h5 = opt.input_img_h5
input_ques_h5 = opt.input_ques_h5
input_json = opt.input_json
early_stop = opt.early_stop
evalall = opt.evalall
file_name = opt.file_name
debug = opt.debug
opt = checkpoint['opt']
opt.start_epoch = checkpoint['epoch']
cur_bs = 5
# opt.data_dir = data_dir
opt.model_path = model_path
opt.input_img_h5 = input_img_h5
opt.input_ques_h5 = input_ques_h5
opt.input_json = input_json
opt.early_stop = early_stop
opt.evalall = evalall
opt.file_name = file_name
opt.debug = debug
if opt.file_name == '':
    print('provide output file name')
    exit(255)

############################## Dataset loading #################################

dataset_val = dl.validate(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                          input_json=opt.input_json,num_val=opt.num_val, data_split='val')

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=5,
                                             shuffle=False, num_workers=int(opt.workers))

vocab_size = dataset_val.vocab_size  # Current value 8964
ques_length = dataset_val.ques_length  # 16
ans_length = dataset_val.ans_length + 1  # 9
his_length = dataset_val.ques_length + dataset_val.ans_length  # 24
itow = dataset_val.itow  # index to word
img_feat_size = 512  # 512

mapping_file = '../script/data/modified_mapped.json'
mapping_data = json.load(open(mapping_file))

################################################################################

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size)
netW = model._netW(vocab_size, opt.ninp, opt.dropout)
netG = _netG(opt.model, vocab_size, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, False)

if opt.evalall:
    netW.load_state_dict(checkpoint['netW_g'])
    netE.load_state_dict(checkpoint['netE_g'])

else:
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
netG.load_state_dict(checkpoint['netG'])

if opt.cuda:
    netW.cuda()
    netE.cuda()
    netG.cuda()

netE.eval()
netW.eval()
netG.eval()

input_img_h5 = '../script/old_data/vdl_img_vgg.h5'
f_image = h5py.File(input_img_h5, 'r')
imgs = f_image['images_train']

batch_size = 120
n = len(mapping_data)

def get_questions_and_history(start, end, mapping_data):
    current_quesitons = mapping_data[start:end]
    question_list = []
    history_list = []
    for i in range(end - start):
        current_dictionary = current_quesitons[i]
        temp_list = []
        temp_list_hist = []
        yes_list = current_dictionary['ques_ans']['ques_yes']
        for yes_dict in yes_list:
            temp_list.append(yes_dict['ques'])
            temp_list_hist.append('')
        no_list = current_dictionary['ques_ans']['ques_no']
        for no_dict in no_list:
            temp_list.append(no_dict['ques'])
            temp_list_hist.append('')
        question_list.append(temp_list)
        history_list.append(temp_list_hist)
    return question_list, history_list

def get_questions_history_tensor(question_data, j, history_data):
    batch_size = len(question_data)
    questions = []
    histories = []
    for i in range(batch_size):
        questions.append(question_list[i][j])
        histories.append(history_list[i][j])

    question_tensor = torch.zeros((16,batch_size),dtype=torch.int64)
    history_tensor = torch.zeros((24,batch_size),dtype=torch.int64)
    for i in range(batch_size):
        question = []
        history = []
        question.append(questions[i])
        history.append(histories[i])
        que, his = get_history_data(question, history)
        question_tensor[:,i:i+1] = que
        history_tensor[:,i:i+1] = his

    return question_tensor, history_tensor

def get_imgs(img_data):
    img = torch.from_numpy(img_data)
    image = img.view(-1, img_feat_size)
    img_input = torch.FloatTensor(opt.batchSize, 49, 512).cuda()
    with torch.no_grad():
        img_input.resize_(image.size()).copy_(image)

    return  img_input

result_all = []

yes_idx = 1188
no_idx = 4059

for i in range(0, n, batch_size):
    start = i
    end = min(n, i + batch_size)
    cur_bs = end - start

    img_data = imgs[start:end]
    question_list, history_list = get_questions_and_history(start, end, mapping_data)  # bs x 6 x 16

    # answer yes no
    ans_fv_np = np.zeros((9, 2, 1), dtype=np.int32)
    ans_fv_np[0] = vocab_size
    ans_fv_np[1, 0] = 1188
    ans_fv_np[1, 1] = 4059

    ans_lv_np = np.zeros((9, 2, 1), dtype=np.int32)
    ans_lv_np[1] = vocab_size
    ans_lv_np[0, 0] = 1188
    ans_lv_np[0, 1] = 4059

    ans_fv = torch.from_numpy(ans_fv_np)
    ans_lv = torch.from_numpy(ans_lv_np)

    ans_fv = ans_fv.expand(9, 2, cur_bs)
    ans_lv = ans_lv.expand(9, 2, cur_bs)

    ans_fv = ans_fv.reshape(9, -1)
    ans_lv = ans_lv.reshape(9, -1)

    save_tmp = [[] for j in range(cur_bs)]

    for j in range(6):

        img_input = get_imgs(img_data)
        ques, his = get_questions_history_tensor(question_list, j, history_list)

        ques_hidden = netE.init_hidden(cur_bs)
        hist_hidden = netE.init_hidden(cur_bs)

        ind = 1

        his_input = torch.LongTensor(his.size()).cuda()
        his_input.copy_(his)

        ques_input = torch.LongTensor(ques.size()).cuda()
        ques_input.copy_(ques)

        ans_input = torch.LongTensor(ans_fv.size()).cuda()
        ans_input.copy_(ans_fv)

        ans_target = torch.LongTensor(ans_lv.size()).cuda()
        ans_target.copy_(ans_lv)

        ques_emb = netW(ques_input, format='index')
        his_emb = netW(his_input, format='index')

        ques_hidden = repackage_hidden_new(ques_hidden, cur_bs)
        hist_hidden = repackage_hidden_new(hist_hidden, his_input.size(1))

        encoder_feat, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                         ques_hidden, hist_hidden, ind)

        _, ques_hidden = netG(encoder_feat.view(1, -1, opt.ninp), ques_hidden)

        ################################################################################

        hidden_replicated = []
        for hid in ques_hidden:
            hidden_replicated.append(hid.view(opt.nlayers, cur_bs, 1, \
                                              opt.nhid).expand(opt.nlayers, cur_bs, 2, opt.nhid).clone().view(
                opt.nlayers, -1, opt.nhid))
        hidden_replicated = tuple(hidden_replicated)

        ans_emb = netW(ans_input, format='index')

        output, _ = netG(ans_emb, hidden_replicated)

        if opt.debug:
            print(output.shape)
            print(output.reshape(9,-1,vocab_size+1)[0,0,yes_idx])
            print(output.reshape(9, -1, vocab_size + 1)[0, 0, no_idx])
            print('-------------------------------')
            print(output.reshape(9, -1, vocab_size + 1)[0, 1, yes_idx])
            print(output.reshape(9, -1, vocab_size + 1)[0, 1, no_idx])

        logprob = - output
        logprob_select = torch.gather(logprob, 1, ans_target.view(-1, 1))

        mask = ans_target.data.eq(0)  # generate the mask
        logprob_select.masked_fill_(mask.view_as(logprob_select), 0)

        prob = logprob_select.view(ans_length, -1, 2).sum(0).view(-1, 2)
        prob_np = prob.cpu().detach().numpy()
        #####################################################################################################3

        # ans_sample = torch.from_numpy(vocab_size*np.ones((cur_bs)))
        #
        # sample_ans_input = torch.LongTensor(1, cur_bs).cuda()
        # sample_ans_input.resize_((1, cur_bs)).fill_(vocab_size)
        #
        # sample_opt = {'beam_size': 1, 'seq_length': 16}
        #
        # seq, seqLogprobs = netG.sample(netW, sample_ans_input, ques_hidden, sample_opt)
        # ans_sample_txt = decode_txt(itow, seq.t())

        for b in range(cur_bs):
            data_dict = {}
            data_dict['yes_prob'] = str(prob_np[b,0])
            data_dict['no_prob'] = str(prob_np[b, 1])
            save_tmp[b].append(data_dict)

    if opt.debug:
        if i>20:
            break

    print('done : {}/{}'.format(i,n))
    result_all += save_tmp

if not opt.debug:
    json.dump(result_all, open(file_name+'.json', 'w'))





