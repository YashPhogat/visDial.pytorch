import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

input_json = '../script/data/visdial_params_demo.json'
word2ind_path = '../script/data/word2ind.json'
f = json.load(open(input_json, 'r'))
itow = f['itow']
word2ind = json.load(open(word2ind_path, 'r'))

ques_length = 16  # Max word length of a question. Current Value is 16
ans_length = 8  # Max word length of answer. Current value is 8
his_length = ques_length + ans_length  # Max word length of question and answer combined. Current value is 16+8 = 24
vocab_size = len(itow) + 1

def word_to_idx(sentence):
    words = word_tokenize(sentence)
    words_out = np.zeros((len(words)),dtype=int)
    for i in range(len(words)):
        if words[i] in word2ind.keys():
            words_out[i] = int(word2ind[words[i]])
        else:
            words_out[i] = len(word2ind.keys())

    return words_out


def get_history_data(quesitons,answers):
    n = len(quesitons)

    # current question
    cur_ques = quesitons[n - 1]
    cur_ques_np = np.zeros((16, 1), dtype=int)
    ques_temp  = word_to_idx(cur_ques)
    cur_ques_np[16-ques_temp.shape[0]:,0] = ques_temp

    #history
    prev_ques = quesitons[0:n - 1]
    prev_ans = answers

    his_np = np.zeros((24,n),dtype=int)

    # #caption
    # caption = np.zeros((24), dtype=int)
    # his_np[:,0] = caption
    caption = prev_ans[0]
    caption_np = word_to_idx(caption)
    nc = caption_np.shape[0]
    his_np[24-nc:,0] = caption_np
    for i in range(len(prev_ques)):
        ques = prev_ques[i]
        ans = prev_ans[i+1]
        ques_np = word_to_idx(ques)
        ans_np = word_to_idx(ans)
        nq = ques_np.shape[0]
        na = ans_np.shape[0]
        his_np[24-nq-na:24-na:,i+1] = ques_np
        his_np[24-na:,i+1] = ans_np

    ques = torch.from_numpy(cur_ques_np)
    his = torch.from_numpy(his_np)

    return ques,his

def generate_ans_from_idx(ids):
    ans = ''
    for i in range(len(ids)):
        idx = str(ids[i].detach().numpy()[0][0])
        if idx in itow.keys():
            word = itow[idx]
            ans+=word
            ans+=' '
        else:
            ans+='.'
            break

    return ans

