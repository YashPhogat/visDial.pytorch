import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F

class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout, img_feat_size):
        super(_netE, self).__init__()

        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.img_embed = nn.Linear(img_feat_size, nhid).cuda()

        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout).cuda()
        self.his_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout).cuda()

        self.Wq_feat_to_emb_for_attention = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wh_feat_to_emb_for_attention = nn.Linear(self.nhid, self.nhid).cuda()
        self.W_attn_by_ques_on_hist = nn.Linear(self.nhid, 1).cuda()

        self.Wq_feat_to_emb_for_attn_img = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wh_feat_to_emb_for_attn_img = nn.Linear(self.nhid, self.nhid).cuda()
        self.Wi_img_emb_to_emb_for_attn_img = nn.Linear(self.nhid, self.nhid).cuda()
        self.W_attention_queshist_img = nn.Linear(self.nhid, 1).cuda()

        self.W_h_mapper = nn.Linear(self.nhid*3, self.nhid)
        self.W_mem_mapper = nn.Linear(self.nhid*3, self.nhid)

        self.fc1 = nn.Linear(self.nhid*3, self.ninp).cuda()

    def forward(self, ques_emb, his_emb, img_raw, ques_hidden, his_hidden, rnd):

        img_emb = F.tanh(self.img_embed(img_raw))

        ques_feat, ques_hidden = self.ques_rnn(ques_emb, ques_hidden)
        ques_feat = ques_hidden[0][0]
        ques_mem = ques_hidden[1][0]

        his_feat, his_hidden = self.his_rnn(his_emb, his_hidden)
        his_feat = his_hidden[0][0]
        his_mem = his_hidden[1][0]

        ques_emb_for_attn_hist = self.Wq_feat_to_emb_for_attention(ques_feat).view(-1, 1, self.nhid)
        his_emb_for_attn_hist = self.Wh_feat_to_emb_for_attention(his_feat).view(-1, rnd, self.nhid)

        atten_emb_for_attn_hist = F.tanh(his_emb_for_attn_hist + ques_emb_for_attn_hist.expand_as(his_emb_for_attn_hist))
        his_atten_weight = F.softmax(self.W_attn_by_ques_on_hist(F.dropout(atten_emb_for_attn_hist, self.d, training=self.training
                                                                           ).view(-1, self.nhid)).view(-1, rnd))

        attn_weighted_hist = torch.bmm(his_atten_weight.view(-1, 1, rnd),
                                        his_feat.view(-1, rnd, self.nhid))

        attn_weighted_hist_mem = torch.bmm(his_atten_weight.view(-1, 1, rnd),
                                        his_mem.view(-1, rnd, self.nhid))

        attn_weighted_hist = attn_weighted_hist.view(-1, self.nhid)
        attn_weighted_hist_mem = attn_weighted_hist_mem.view(-1, self.nhid)

        ques_emb_for_attn_img = self.Wq_feat_to_emb_for_attn_img(ques_feat).view(-1, 1, self.nhid)
        his_emb_for_attn_img = self.Wh_feat_to_emb_for_attn_img(attn_weighted_hist).view(-1, 1, self.nhid)
        img_emb_for_attn_img = self.Wi_img_emb_to_emb_for_attn_img(img_emb).view(-1, 49, self.nhid)

        atten_emb_for_attn_img = F.tanh(img_emb_for_attn_img + ques_emb_for_attn_img.expand_as(img_emb_for_attn_img) + \
                                    his_emb_for_attn_img.expand_as(img_emb_for_attn_img))

        img_atten_weight = F.softmax(self.W_attention_queshist_img(F.dropout(atten_emb_for_attn_img, self.d, training=self.training
                                                                             ).view(-1, self.nhid)).view(-1, 49))

        attn_weighted_img_feat = torch.bmm(img_atten_weight.view(-1, 1, 49),
                                        img_emb.view(-1, 49, self.nhid))

        concat_feat = torch.cat((ques_feat, attn_weighted_hist.view(-1, self.nhid), \
                                 attn_weighted_img_feat.view(-1, self.nhid)),1)

        concat_mem = torch.cat((ques_mem, attn_weighted_hist_mem.view(-1, self.nhid), \
                                 attn_weighted_img_feat.view(-1, self.nhid)),1)

        encoder_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))

        mapped_feat = F.tanh(self.W_h_mapper(F.dropout(concat_feat)))
        mapped_feat = mapped_feat.unsqueeze(0)

        mapped_mem = F.tanh(self.W_mem_mapper(F.dropout(concat_mem)))
        mapped_mem = mapped_mem.unsqueeze(0)

        return encoder_feat, (mapped_feat, mapped_mem)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda(),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
