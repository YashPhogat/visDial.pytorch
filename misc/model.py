import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from misc.share_Linear import share_Linear
from texttable import Texttable
from getch import pause

class _netW(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(_netW, self).__init__()
        self.word_embed = nn.Embedding(ntoken+1, ninp).cuda()
        self.Linear = share_Linear(self.word_embed.weight).cuda()
        self.init_weights()
        self.d = dropout

    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, format ='index'):
        if format == 'onehot':
            out = F.dropout(self.Linear(input), self.d, training=self.training)
        elif format == 'index':
            out = F.dropout(self.word_embed(input), self.d, training=self.training)

        return out

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda(),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class _netD(nn.Module):
    """
    Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer.
    """
    def __init__(self, rnn_type, ninp, nhid, nlayers, ntoken, dropout):
        super(_netD, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.ninp = ninp
        self.d = dropout

        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers).cuda()
        self.W1 = nn.Linear(self.nhid, self.nhid).cuda()
        self.W2 = nn.Linear(self.nhid, 1).cuda()
        self.fc = nn.Linear(nhid, ninp).cuda()

    def forward(self, input_feat, idx, hidden, vocab_size):

        output, _ = self.rnn(input_feat, hidden)
        mask = idx.data.eq(0)  # generate the mask
        mask[idx.data == vocab_size] = 1 # also set the last token to be 1
        if isinstance(input_feat, Variable):
            mask = Variable(mask, volatile=input_feat.volatile).cuda()

        # Doing self attention here.
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.d, training=self.training)).view(idx.size())
        atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten.t()).view(-1,1,idx.size(0))
        feat = torch.bmm(weight, output.transpose(0,1)).view(-1,self.nhid)
        feat = F.dropout(feat, self.d, training=self.training)
        transform_output = F.tanh(self.fc(feat))

        return transform_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda(),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class  LMCriterion(nn.Module):

    def __init__(self):
        super(LMCriterion, self).__init__()

    def forward(self, input, target):
        logprob_select = torch.gather(input, 1, target)

        mask = target.data.gt(0)  # generate the mask
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile).cuda()
        
        out = torch.masked_select(logprob_select, mask)

        loss = -torch.sum(out) # get the average loss.
        return loss


class mixture_of_softmaxes(torch.nn.Module):
    """
    Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (ICLR 2018)    
    """
    def __init__(self, nhid, n_experts, ntoken):

        super(mixture_of_softmaxes, self).__init__()
        
        self.nhid=nhid
        self.ntoken=ntoken
        self.n_experts=n_experts
        
        self.prior = nn.Linear(nhid, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhid, n_experts*nhid), nn.Tanh())
        self.decoder = nn.Linear(nhid, ntoken)
   
    def forward(self, x):
        
        latent = self.latent(x)
        logit = self.decoder(latent.view(-1, self.nhid))

        prior_logit = self.prior(x).view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken)).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)
        
        return prob

class nPairLoss(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """
    def __init__(self, ninp, margin, alpha_norm=0.1, sigma=1.0, sample_each=5,
                 debug = False, log_iter=5):
        super(nPairLoss, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)
        self.alpha_norm = alpha_norm
        self.sigma = sigma
        self.debug = debug
        self.iter = 0
        self.log_iter = log_iter
        self.sample_each = sample_each

    def forward(self, feat, sampled_ans, num_individual, fake=None, fake_diff_mask=None):
        batch_size = feat.size(0)

        mask_for_samples = num_individual.lt(self.sample_each)
        batch_level_ans_mask = torch.sum(mask_for_samples,dim=1)

        batch_level_ans_mask = torch.logical_not(batch_level_ans_mask)

        if torch.sum(batch_level_ans_mask)<sampled_ans.shape[0]:
            print('Daav thyo')
            print(torch.sum(batch_level_ans_mask))

        batch_level_ans_mask = batch_level_ans_mask.reshape(batch_size,1,1)
        batch_level_feat_mask = batch_level_ans_mask.reshape(batch_size,1)

        batch_size = torch.sum(batch_level_ans_mask)
        final_batch_level_ans_mask = batch_level_ans_mask.expand_as(sampled_ans)

        new_sampled_ans = torch.masked_select(sampled_ans,final_batch_level_ans_mask).reshape(batch_size,sampled_ans.shape[1],sampled_ans.shape[2])

        # batch_size = new_sampled_ans.shape[0]
        contra_ans_emb = new_sampled_ans[:, 0: self.sample_each, :]
        entail_ans_emb = new_sampled_ans[:, self.sample_each:2*self.sample_each, :]
        neutra_ans_emb = new_sampled_ans[:, 2*self.sample_each:3*self.sample_each, :]

        final_batch_level_feat_mask = batch_level_feat_mask.expand_as(feat)
        feat = torch.masked_select(feat,final_batch_level_feat_mask).reshape(batch_size,feat.shape[1])
        feat = feat.view(-1, self.ninp, 1)

        contra_scores = torch.bmm(contra_ans_emb, feat)
        entail_scores = torch.bmm(entail_ans_emb, feat)
        neutra_scores = torch.bmm(neutra_ans_emb, feat)

        contra_scores_permute = contra_scores.permute(0,2,1)
        neutra_scores_permute = neutra_scores.permute(0,2,1)

        pair_wise_score_diff_ec = entail_scores.expand(batch_size, self.sample_each, self.sample_each) - contra_scores_permute.expand(batch_size, self.sample_each, self.sample_each)
        pair_wise_score_diff_en = entail_scores.expand(batch_size, self.sample_each, self.sample_each) - neutra_scores_permute.expand(batch_size, self.sample_each, self.sample_each)
        pair_wise_score_diff_nc = neutra_scores.expand(batch_size, self.sample_each, self.sample_each) - contra_scores_permute.expand(batch_size, self.sample_each, self.sample_each)

        norm_loss = feat.norm() + new_sampled_ans.norm()

        if self.debug:
            if self.iter % self.log_iter == 0:
                print('in debug mode')
                # print('---------------- Score difference: --------------')
                # rows = [['data_'+str(i) for i in range(batch_size)]]
                # pair_wise_score_diff_np = pair_wise_score_diff.cpu().detach().numpy()
                # wrong_scores_np = wrong_dis.cpu().detach().numpy()
                # right_scores_np = right_dis.cpu().detach().numpy()
                #
                # for j in range(num_wrong):
                #     row = []
                #     for i in range(batch_size):
                #         row.append('%.4f | %.4f | %.4f' % (np.around(right_scores_np[i][0][0], 4), np.around(wrong_scores_np[i][j][0], 4),
                #                                      np.round(pair_wise_score_diff_np[i][j], 4)))
                #     rows.append(row)
                # st = Texttable()
                # st.add_rows(rows)
                # print(st.draw())
                # print('----------------Probabilities------------------')
                # print(probs.cpu().detach().numpy())
                # print('----------------One hot------------------------')
                # print(one_hot_probs.cpu().detach().numpy())
                # print('----------------dist_summary-------------------')
                # print(dist_summary.cpu().detach().numpy())
                # print('----------------smooth_dist_summary------------')
                # print(smooth_dist_summary.cpu().detach().numpy())
                # pause()

        loss = torch.log(1/1+torch.exp(-self.sigma*pair_wise_score_diff_ec)) + \
               torch.log(1/1+torch.exp(-self.sigma*pair_wise_score_diff_en)) + \
               torch.log(1/1+torch.exp(-self.sigma*pair_wise_score_diff_nc))
        loss = torch.sum(loss)
        loss = -loss
        total_loss = loss + self.alpha_norm*norm_loss
        total_loss = total_loss/batch_size
        return total_loss

class G_loss(nn.Module):
    """
    Generator loss:
    minimize right feature and fake feature L2 norm.
    maximinze the fake feature and wrong feature.
    """
    def __init__(self, ninp):
        super(G_loss, self).__init__()
        self.ninp = ninp

    def forward(self, feat, right, fake):

        #num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        #wrong_dis = torch.bmm(wrong, feat)
        #batch_wrong_dis = torch.bmm(batch_wrong, feat)
        fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)

        fake_score = torch.exp(right_dis - fake_dis)
        loss_fake = torch.sum(torch.log(fake_score + 1))

        loss_norm = feat.norm() + fake.norm() + right.norm()
        loss = (loss_fake + 0.1 * loss_norm) / batch_size

        return loss, loss_fake.data.item()/batch_size


class gumbel_sampler(nn.Module):
    def __init__(self):
        super(gumbel_sampler, self).__init__()

    def forward(self, input, noise, temperature=0.5):

        eps = 1e-20
        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()
        y = (input + noise) / temperature
        y = F.softmax(y)

        max_val, max_idx = torch.max(y, y.dim()-1)
        y_hard = y == max_val.view(-1,1).expand_as(y)
        y = (y_hard.float() - y).detach() + y

        # log_prob = input.gather(1, max_idx.view(-1,1)) # gather the logprobs at sampled positions

        return y, max_idx.view(1, -1)#, log_prob

class AxB(nn.Module):
    def __init__(self, nhid):
        super(AxB, self).__init__()
        self.nhid = nhid

    def forward(self, nhA, nhB):
        mat = torch.bmm(nhB.view(-1, 100, self.nhid), nhA.view(-1,self.nhid,1))
        return mat.view(-1,100)
