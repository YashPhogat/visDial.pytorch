import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt


class train(data.Dataset) :  # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, num_val, data_split,
                 input_probs='../script/data/visdial_data_prob.h5',entail_thers = 0.5, contra_thres = 0.9, sample_each = 5) :
        #This is the number of images for which we have copied the new vgg features to the parallely
        #accessible h5 file. DO NOT CHANGE THIS!!!
        self.TOTAL_VALID_IMAGES = 8000
        self.entail_thres = entail_thers
        self.contra_thres = contra_thres
        self.sample_each = sample_each
        self.total_sample = 3*self.sample_each
        print(h5py.version.info)
        print('DataLoader loading: %s' % data_split)
        print('Loading image feature from %s' % input_img_h5)

        if data_split == 'test' :
            split = 'val'
        else :
            split = 'train'  # train and val split both corresponding to 'train'

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_' + split]

        self.f_image = h5py.File(input_img_h5, 'r')
        self.imgs = self.f_image['images_' + split]

        # get the data split.
        total_num = self.TOTAL_VALID_IMAGES
        if data_split == 'train' :
            s = 0
            e = total_num - num_val
        elif data_split == 'val' :
            s = total_num - num_val
            e = total_num
        else :
            s = 0
            e = total_num

        self.img_info = self.img_info[s :e]

        print('%s number of data: %d' % (data_split, e - s))
        # load the data.

        # f.close()

        print('Loading txt from %s' % input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_' + split][s :e]
        self.ans = f['ans_' + split][s :e]
        self.cap = f['cap_' + split][s :e]

        self.ques_len = f['ques_len_' + split][s :e]
        self.ans_len = f['ans_len_' + split][s :e]
        self.cap_len = f['cap_len_' + split][s :e]

        self.ans_ids = f['ans_index_' + split][s :e]
        self.opt_ids = f['opt_' + split][s :e]
        self.opt_list = f['opt_list_' + split][:]
        self.opt_len = f['opt_len_' + split][:]
        f.close()

        self.ques_length = self.ques.shape[2]  # Max word length of a question. Current Value is 16
        self.ans_length = self.ans.shape[2]  # Max word length of answer. Current value is 8
        self.his_length = self.ques_length + self.ans_length  # Max word length of question and answer combined. Current value is 16+8 = 24
        self.vocab_size = len(self.itow) + 1

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.total_qa_pairs = 10

        f = h5py.File(input_probs, 'r')
        opt_probs_temp = f['opt_train'][s:e]
        total_images = e-s
        self.opt_probs = self._process_probs(opt_probs_temp, total_images)
        self.opt_tags,  self.tag_frequency = self.get_tags(self.opt_probs, self.entail_thres, self.contra_thres)
        f.close()

    def _process_probs(self, long_probs, total_images):
        probs = np.ndarray(shape=(total_images, 10, 100, 3), dtype=np.float)
        magic = int(10000)
        div = long_probs//magic
        rem = long_probs%magic
        probs[:, :, :, 2] = rem/(float(magic))
        probs[:, :, :, 1] = div/(float(magic))
        probs[:, :, :, 0] = 1 -probs[:, :, :, 1] - probs[:, :, :, 2]
        return probs

    def get_tags(self, probs, entail_thres, contra_thres):
        entail_mask = np.zeros(probs.shape, dtype=np.bool)
        entail_mask[:, :, :, 1] = True
        entail_thres_mask = probs > entail_thres
        entail_final_mask = entail_mask * entail_thres_mask
        entail_final_mask_sum = np.sum(entail_final_mask, axis=-1)

        contra_mask = np.zeros(probs.shape, dtype=np.bool)
        contra_mask[:, :, :, 0] = True
        contra_thres_mask = probs > contra_thres
        contra_final_mask = contra_mask * contra_thres_mask
        contra_final_mask_sum = 2 * np.sum(contra_final_mask, axis=-1)

        tags = np.zeros(shape=(probs.shape[0], probs.shape[1], probs.shape[2]), dtype=np.int8)

        tags = entail_final_mask_sum + contra_final_mask_sum
        tags = 2 - tags

        frequency = np.zeros(shape=(probs.shape[0],probs.shape[1],probs.shape[3]),dtype=np.int32)
        frequency[:,:,0] = np.sum(tags==0,axis=-1)
        frequency[:, :, 1] = np.sum(tags == 1, axis=-1)
        frequency[:, :, 2] = np.sum(tags == 2, axis=-1)
        return tags, frequency

    def __getitem__(self, index) :
        # get the image
        img = torch.from_numpy(self.imgs[index])

        # get the history
        #Format of one row of his:
        #0 0 .. 0 0 q q q q a a a a
        #leading zero - (question words - answer words) OR leading zero - caption words
        his = np.zeros((self.total_qa_pairs, self.his_length))
        his[0, self.his_length - self.cap_len[index] :] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.total_qa_pairs, self.ques_length))
        ques_trailing_zeros = np.zeros((self.total_qa_pairs, self.ques_length))

        opt_ans_vocab_first = np.zeros((self.total_qa_pairs, self.total_sample, self.ans_length + 1))
        opt_ans_len = np.zeros((self.total_qa_pairs, self.total_sample))

        ans_idx = np.zeros((self.total_qa_pairs))
        opt_ans_idx = np.zeros((self.total_qa_pairs, self.total_sample))
        opt_selected_probs = np.zeros((self.total_qa_pairs, self.total_sample, 3))

        num_individual = np.zeros((self.total_qa_pairs,3), dtype=np.int32)
        for i in range(self.total_qa_pairs) :
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]

            qa_len = q_len + a_len

            if i + 1 < self.total_qa_pairs :
                his[i + 1, self.his_length - qa_len :self.his_length - a_len] = self.ques[index, i, :q_len]
                his[i + 1, self.his_length - a_len :] = self.ans[index, i, :a_len]

            ques[i, self.ques_length - q_len :] = self.ques[index, i, :q_len]

            ques_trailing_zeros[i, :q_len] = self.ques[index, i, :q_len]

            ########################################################################

            tag_indices = np.argsort(self.opt_tags[index,i])
            total_num_contra = self.tag_frequency[index,i,0]
            total_num_entail = self.tag_frequency[index, i, 1]
            total_num_neutra = self.tag_frequency[index, i, 2]
            num_contra = min(self.sample_each, total_num_contra)
            num_entail = min(self.sample_each, total_num_entail)
            num_neutra = self.total_sample - num_contra - num_entail

            num_individual[i,0] = num_contra
            num_individual[i,1] = num_entail
            num_individual[i,2] = num_neutra

            contra_list = tag_indices[:total_num_contra]
            random.shuffle(contra_list)
            entail_list = tag_indices[total_num_contra:total_num_contra+total_num_entail]
            random.shuffle(entail_list)
            neutra_list = tag_indices[-total_num_neutra:]
            random.shuffle(neutra_list)

            entail_list[0] = self.ans_ids[index,i]
            opt_ids_temp = self.opt_ids[index, i]
            opt_ids = []
            for j in range(num_contra):
                opt_ids.append(opt_ids_temp[contra_list[j]])
            for j in range(num_entail):
                opt_ids.append(opt_ids_temp[entail_list[j]])
            for j in range(num_neutra):
                opt_ids.append(opt_ids_temp[neutra_list[j]])

            ########################################################################

            for j in range(self.total_sample) :
                ids = opt_ids[j]
                opt_ans_idx[i, j] = ids

                opt_len = self.opt_len[ids]

                opt_ans_len[i, j] = opt_len
                opt_ans_vocab_first[i, j, :opt_len] = self.opt_list[ids, :opt_len]
                opt_ans_vocab_first[i, j, opt_len] = self.vocab_size

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)


        ques_trailing_zeros = torch.from_numpy(ques_trailing_zeros)

        opt_ans_len = torch.from_numpy(opt_ans_len)
        opt_ans_vocab_first = torch.from_numpy(opt_ans_vocab_first)

        num_individual = torch.from_numpy(num_individual)

        return img, his, ques, ques_trailing_zeros, \
               opt_ans_vocab_first, opt_ans_len, num_individual

    def __len__(self) :
        return self.ques.shape[0]


class validate(data.Dataset) :  # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, num_val, data_split) :
        # This is the number of images for which we have copied the new vgg features to the parallely
        # accessible h5 file. DO NOT CHANGE THIS!!!
        TOTAL_VALID_TEST_IMAGES = 40000
        TOTAL_VALID_TRAIN_IMAGES = 82000
        print('DataLoader loading: %s' % data_split)
        print('Loading image feature from %s' % input_img_h5)
        total_num = 0
        if data_split == 'test' :
            total_num = TOTAL_VALID_TEST_IMAGES
            split = 'val'
        else :
            total_num = TOTAL_VALID_TRAIN_IMAGES
            split = 'train'  # train and val split both corresponding to 'train'

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_' + split]

        self.f_image = h5py.File(input_img_h5, 'r')
        self.imgs = self.f_image['images_' + split]

        # get the data split.

        if data_split == 'train' :
            e = total_num - num_val
        elif data_split == 'val' :
            s = total_num - num_val
            e = total_num
        else :
            s = 0
            e = total_num

        self.img_info = self.img_info[s :e]
        print('%s number of data: %d' % (data_split, e - s))

        # load the data.

        print('Loading txt from %s' % input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_' + split][s :e]
        self.ans = f['ans_' + split][s :e]
        self.cap = f['cap_' + split][s :e]

        self.ques_len = f['ques_len_' + split][s :e]
        self.ans_len = f['ans_len_' + split][s :e]
        self.cap_len = f['cap_len_' + split][s :e]

        self.ans_ids = f['ans_index_' + split][s :e]
        self.opt_ids = f['opt_' + split][s :e]
        self.opt_list = f['opt_list_' + split][:]
        self.opt_len = f['opt_len_' + split][:]
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow) + 1

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.total_qa_pairs = 10

    def __getitem__(self, index) :

        # get the image
        img_id = self.img_info[index]['imgId']
        img = torch.from_numpy(self.imgs[index])
        # get the history
        his = np.zeros((self.total_qa_pairs, self.his_length))
        his[0, self.his_length - self.cap_len[index] :] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.total_qa_pairs, self.ques_length))
        ans_vocab_first = np.zeros((self.total_qa_pairs, self.ans_length + 1))
        ans_vocab_last = np.zeros((self.total_qa_pairs, self.ans_length + 1))
        ques_trailing_zeros = np.zeros((self.total_qa_pairs, self.ques_length))

        opt_ans_vocab_first = np.zeros((self.total_qa_pairs, 100, self.ans_length + 1))
        ans_idx = np.zeros(self.total_qa_pairs)
        opt_ans_vocab_last = np.zeros((self.total_qa_pairs, 100, self.ans_length + 1))

        ans_len = np.zeros((self.total_qa_pairs))
        opt_ans_len = np.zeros((self.total_qa_pairs, 100))

        for i in range(self.total_qa_pairs) :
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i + 1 < self.total_qa_pairs :
                ques_ans = np.concatenate([self.ques[index, i, :q_len], self.ans[index, i, :a_len]])
                his[i + 1, self.his_length - qa_len :] = ques_ans

            ques[i, self.ques_length - q_len :] = self.ques[index, i, :q_len]
            ques_trailing_zeros[i, :q_len] = self.ques[index, i, :q_len]
            ans_vocab_first[i, 1 :a_len + 1] = self.ans[index, i, :a_len]
            ans_vocab_first[i, 0] = self.vocab_size

            ans_vocab_last[i, :a_len] = self.ans[index, i, :a_len]
            ans_vocab_last[i, a_len] = self.vocab_size

            ans_idx[i] = self.ans_ids[index, i]  # since python start from 0
            opt_ids = self.opt_ids[index, i]  # since python start from 0
            ans_len[i] = self.ans_len[index, i]

            for j, ids in enumerate(opt_ids) :
                opt_len = self.opt_len[ids]
                opt_ans_vocab_first[i, j, 1 :opt_len + 1] = self.opt_list[ids, :opt_len]
                opt_ans_vocab_first[i, j, 0] = self.vocab_size

                opt_ans_vocab_last[i, j, :opt_len] = self.opt_list[ids, :opt_len]
                opt_ans_vocab_last[i, j, opt_len] = self.vocab_size
                opt_ans_len[i, j] = opt_len

        opt_ans_vocab_first = torch.from_numpy(opt_ans_vocab_first)
        opt_ans_vocab_last = torch.from_numpy(opt_ans_vocab_last)
        ans_idx = torch.from_numpy(ans_idx)

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans_vocab_first = torch.from_numpy(ans_vocab_first)
        ans_vocab_last = torch.from_numpy(ans_vocab_last)
        ques_trailing_zeros = torch.from_numpy(ques_trailing_zeros)

        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)

        return img, his, ques, ans_vocab_first, ans_vocab_last, ques_trailing_zeros, opt_ans_vocab_first, \
               opt_ans_vocab_last, ans_idx, ans_len, opt_ans_len, img_id

    def __len__(self) :
        return self.ques.shape[0]
