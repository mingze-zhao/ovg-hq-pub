"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Copyright (c) 2022 WonJun Moon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from lighthouse.common.utils.basic_utils import load_jsonl, l2_normalize_np_array
from lighthouse.common.utils.tensor_utils import pre_pad_sequences_1d, pad_sequences_1d
from lighthouse.common.utils.span_utils import span_xx_to_cxw
from torchtext import vocab
import torch.nn as nn
torch.set_num_threads(1)
import h5py
import pickle
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import copy
from pyinstrument import Profiler
import time
logger = logging.getLogger(__name__)

class StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """
    def __init__(self, dset_name, domain, data_path, v_feat_dirs, a_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state", v_feat_types="clip", a_feat_types="pann", 
                 max_q_l=32, max_v_l=75, max_a_l=75, ctx_mode="video", clip_len=2,
                 max_windows=5, span_loss_type="l1", load_labels=True, segment_size=16, use_online=False, debug=False, anchor_windows=[], pos_threshold=None, label_file_path=None, subset=None, opt=None, training=False):
        self.dset_name = dset_name
        self.domain = domain
        self.data_path = data_path
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.a_feat_dirs = a_feat_dirs \
            if isinstance(a_feat_dirs, list) else [a_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_types = v_feat_types
        self.a_feat_types = a_feat_types
        
        if max_v_l == -1:
            max_v_l = 100000000
        if max_a_l == -1:
            max_a_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.max_a_l = max_a_l
        
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.use_audio = "audio" in ctx_mode
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.load_labels = load_labels
        self.use_online = use_online
        self.anchor_windows=anchor_windows
        self.debug = debug
        self.pos_threshold = pos_threshold
        self.label_file_path = label_file_path
        self.subset = subset
        # data
        self.data = self.load_data()
        self.qid_data_dict = {str(d['qid']): d for d in self.data}
        self.minimal_unit = opt.minimal_unit
        self.encoder_layer_type = opt.encoder_layer_type
        self.training = training

        if self.dset_name == 'tvsum' or self.dset_name == 'youtube_highlight':
            new_data = []
            for d in self.data:
                if d['domain'] == self.domain:
                    new_data.append(d)
            self.data = new_data

        self.use_glove = 'glove' in q_feat_dir
        if self.use_glove:
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)
        
        if use_online:
            self.segment_size = segment_size
            self.sequence_length = opt.sequence_length
            self._makeInputSeq() #返回整个Dataset的所有segment,[Nseg]
            self._get_video_query_feat_dict()
            self._loadPropLabel(self.label_file_path.format(self.subset))

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        return datalist

    def __len__(self):
        if self.use_online:
            return len(self.inputs)
        return len(self.data) 

    def __getitem__(self, index):
        if self.use_online:
            qid, vid, st, ed, data_idx, line_idx = self.inputs[index]        
            if st >= 0:
                query_feat, video_feat = self._get_base_data(qid,vid,st,ed)
            else :
                query_feat, video_feat = self._get_base_data(qid,vid,0,ed)
            if self.minimal_unit == "clip":       
                cls_label=torch.Tensor(self.cls_label[data_idx])
                reg_label=torch.Tensor(self.reg_label[data_idx])
            elif self.minimal_unit == "window":
                if self.training:
                    cls_label=torch.Tensor(self.cls_label[data_idx:data_idx+self.sequence_length])
                    reg_label=torch.Tensor(self.reg_label[data_idx:data_idx+self.sequence_length])
                else:
                    cls_label=torch.Tensor(self.cls_label[data_idx:data_idx+ed])
                    reg_label=torch.Tensor(self.reg_label[data_idx:data_idx+ed])
                    # 下面这些视频label有问题，所以这里特殊处理一下
                    if cls_label.shape[0] != video_feat.shape[0]:
                    # if vid in ["KQyJtq52Jcw_660.0_810.0", "Z-L9RjWBTHg_360.0_510.0", "rNPSRSs3reQ_60.0_210.0", "IAbAn-MkMH8_510.0_660.0", "J4pIK7YehhQ_60.0_210.0","j7rJstUseKg_360.0_510.0","j7rJstUseKg_60.0_210.0","-Oc6gSWB_HA_60.0_210.0","G60-kHBEeZA_60.0_210.0","Ok-M_V_h-eY_210.0_360.0","Ok-M_V_h-eY_360.0_510.0","Ok-M_V_h-eY_60.0_210.0","S73Z-nM0GQE_60.0_210.0","S73Z-nM0GQE_210.0_360.0","S73Z-nM0GQE_360.0_510.0","S73Z-nM0GQE_660.0_810.0","S73Z-nM0GQE_510.0_660.0","6JnES9tDKy8_210.0_360.0","uSAGSbauHBs_510.0_660.0"]: # 这个视频的label有问题，所以这里特殊处理一下
                        cls_label=torch.Tensor(self.cls_label[data_idx:data_idx+ed-1])
                        reg_label=torch.Tensor(self.reg_label[data_idx:data_idx+ed-1])
            meta = self.data[line_idx]
            model_inputs = dict()
            model_inputs["query_feat"] = query_feat
            model_inputs["video_feat"] = video_feat
            model_inputs["cls_label"] = cls_label
            model_inputs["reg_label"] = reg_label
            ctx_l = len(model_inputs["video_feat"])
            if self.use_tef:
                tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
                tef_ed = tef_st + 1.0 / ctx_l
                tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
                if self.use_video:
                    model_inputs["video_feat"] = torch.cat(
                        [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
                else:
                    model_inputs["video_feat"] = tef
            return dict(meta=meta, model_inputs=model_inputs)
        else:
            meta = self.data[index]

            model_inputs = dict()

            if self.use_glove:
                model_inputs["query_feat"] = self.get_query(meta["query"])
            else:
                model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)

            if self.use_video:
                model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
                ctx_l = len(model_inputs["video_feat"])
            else:
                ctx_l = self.max_v_l

            if self.use_audio:
                assert self.a_feat_types is not None, f"use_audio is {self.use_audio}, but a_feat_types is {self.a_feat_types}."
                model_inputs["audio_feat"] = self._get_audio_feat_by_vid(meta["vid"])
                ctx_l_a = len(model_inputs["audio_feat"])
                # Sometimes, audio features is longer than video features because the length of video is not necessarily 2:30.
                if ctx_l < ctx_l_a:
                    model_inputs["audio_feat"] = model_inputs["audio_feat"][:ctx_l]
                    ctx_l_a = ctx_l
                elif ctx_l > ctx_l_a:
                    model_inputs["video_feat"] = model_inputs["video_feat"][:ctx_l_a] # TODO: Sometimes, audio length is not equal to video length.
                    ctx_l = ctx_l_a
            else:
                ctx_l_a = self.max_a_l

            if self.use_tef:
                tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
                tef_ed = tef_st + 1.0 / ctx_l
                tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
                if self.use_video:
                    model_inputs["video_feat"] = torch.cat(
                        [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
                else:
                    model_inputs["video_feat"] = tef

            if self.load_labels:
                if self.dset_name == 'tvsum':
                    model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                    meta_label = meta["label"]
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                                self.get_saliency_labels_all_tvsum(meta_label, ctx_l)
                    if len(model_inputs["saliency_all_labels"]) != len(model_inputs["video_feat"]):
                        model_inputs["video_feat"] = model_inputs["video_feat"][:len(model_inputs["saliency_all_labels"])]
                
                elif self.dset_name == 'youtube_highlight':
                    model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                    meta_label = meta['label']
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                                self.get_saliency_labels_all_youtube(meta_label, ctx_l)
                
                else:
                    model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)
                    model_inputs["pos_mask"] = self.get_pos_mask(meta, ctx_l) # necessary for TR-DETR. If you dont use it, ignore.

                    if 'qvhighlight' in self.dset_name:
                        if "subs_train" in self.data_path: # for pretraining
                            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                                self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)
                        else:
                            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                                self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)                        
                    
                    elif self.dset_name in ['charades', 'tacos', 'activitynet']:
                        model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)
                    else:
                        raise NotImplementedError()

            return dict(meta=meta, model_inputs=model_inputs)

    def get_pos_mask(self, meta, ctx_l):
        # necessary only for TR-DETR: model_inputs["pos_mask"]
        if 'relevant_clip_ids' in meta:
            pos_idx = torch.tensor(meta['relevant_clip_ids'])
        else:
            # TODO: Implemented pos_mask for MR/HD tasks for TR-DETR, but I could not reproduce the reported scores
            clip_start_ind = math.floor(meta["relevant_windows"][0][0] / self.clip_len)
            clip_end_ind = math.ceil(meta["relevant_windows"][0][1] / self.clip_len)
            if clip_start_ind == clip_end_ind:
                clip_end_ind += 1 # to avoid a bug
            pos_idx = torch.tensor([i for i in range(clip_start_ind, clip_end_ind)])

        mask = torch.zeros_like(torch.ones(ctx_l))
        if pos_idx.max() >= len(mask):
            new_mask = torch.zeros_like(torch.ones(pos_idx.max()+1 ))
            new_mask[pos_idx] = 1
            new_mask[:len(mask)] = mask
            mask = new_mask
        else:
            mask[pos_idx] = 1

        if self.dset_name in ['charades', 'tacos', 'activitynet']:
            mask = mask[:ctx_l]

        return mask

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l)) # to fix bugs / works..?
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed+1] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            # if rel_clip_ids[idx] == ctx_l:
            #     print(rel_clip_ids[idx], ctx_l)
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        
        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_youtube(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        # Youtube-hl only have binary score
        agg_scores = np.array(labels)[:, 0] # (L, 1) --> (L, )
        score_array = agg_scores * 1
        
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum' or self.dset_name == 'youtube_highlight':
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)
            return torch.from_numpy(q_feat['token']) if self.dset_name == 'tvsum' else torch.from_numpy(q_feat['last_hidden_state'])

        else:
            if self.dset_name == 'tacos':
                q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            elif "subs_train" in self.data_path: # for pretrain
                vid = "_".join(qid.split("_")[:-1])
                subid = qid.split("_")[-1]
                q_feat_path = join(self.q_feat_dir, f"{vid}/{subid}.npz")
            else:
                q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")

            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            q_feat = l2_normalize_np_array(q_feat)
            return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            if self.dset_name == 'tvsum' and 'i3d' in _feat_dir:
                rgb_path = join(_feat_dir, f"{vid}_rgb.npy")
                opt_path = join(_feat_dir, f"{vid}_opt.npy")
                rgb_feat = np.load(rgb_path)[:self.max_v_l].astype(np.float32)
                opt_feat = np.load(opt_path)[:self.max_v_l].astype(np.float32)
                _feat = np.concatenate([rgb_feat, opt_feat], axis=-1)
                _feat = l2_normalize_np_array(_feat) # normalize?
                v_feat_list.append(_feat)
            else:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
        
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)
    
    def _get_audio_feat_by_vid(self, vid):
        a_feat_list = []
        for _feat_dir in self.a_feat_dirs:
            if self.dset_name == 'qvhighlight':
                if self.a_feat_types == "clap":
                    _feat_path = join(_feat_dir, f"{vid}.npz")
                    _feat = np.load(_feat_path)["features"][:self.max_a_l].astype(np.float32)
                elif self.a_feat_types == "pann":
                    _feat_path = join(_feat_dir, f"{vid}.npy")
                    _feat = np.load(_feat_path)[:self.max_a_l].astype(np.float32)
                else:
                    raise NotImplementedError()
                _feat = l2_normalize_np_array(_feat) # normalize?
                a_feat_list.append(_feat)
            else:
                raise NotImplementedError()
        
        # some features are slightly longer than the others
        min_len = min([len(e) for e in a_feat_list])
        a_feat_list = [e[:min_len] for e in a_feat_list]
        a_feat = np.concatenate(a_feat_list, axis=1)
        return torch.from_numpy(a_feat)  # (Lv, D)

    def _makeInputSeq(self):
        data_idx=0
        self.inputs=[]
        if self.training:
            for index in range(0,len(self.data)):
                qid=self.data[index]['qid']
                vid=self.data[index]['vid']    
                duration = self.data[index]['duration']
                duration /= self.clip_len
                if self.minimal_unit == "clip":
                    for i in range(1, int(duration)+1):
                        st = i-self.segment_size
                        ed = i
                        self.inputs.append([qid,vid,st,ed,data_idx,index])
                        data_idx+=1
                elif self.minimal_unit == "window":
                    for i in range(1, int(duration)+ 2 - self.sequence_length):
                        st = i-self.segment_size
                        ed = i + self.sequence_length - 1
                        self.inputs.append([qid,vid,st,ed,data_idx,index])
                        if i == (int(duration)+ 1 - self.sequence_length):
                            data_idx+=self.sequence_length
                        else:
                            data_idx+=1
        else:
            for index in range(0,len(self.data)):
                qid=self.data[index]['qid']
                vid=self.data[index]['vid']    
                duration = self.data[index]['duration']
                duration /= self.clip_len
                if self.minimal_unit == "clip":
                    for i in range(1, int(duration)+1):
                        st = i-self.segment_size
                        ed = i
                        self.inputs.append([qid,vid,st,ed,data_idx,index])
                        data_idx+=1
                elif self.minimal_unit == "window":
                        st = 1-self.segment_size
                        ed = int(duration)
                        self.inputs.append([qid,vid,st,ed,data_idx,index])
                        data_idx+= int(duration)
        print ("subset sample numbers: %d" %(len(self.inputs)))

    def _get_video_query_feat_dict(self):
        self.video_feat_dict = {}
        self.query_feat_dict = {}
        flag_v = torch.zeros(75,2816) 
        flag_q = torch.zeros(10,512)
        print("--------load video feature----------")
        for i in tqdm(range(len(self.data))): 
            video_name=self.data[i]['vid']
            if video_name not in self.video_feat_dict:
                if self.debug:
                    self.video_feat_dict[video_name] = flag_v
                else:
                    self.video_feat_dict[video_name]=self._get_video_feat_by_vid(video_name)
                
        print("--------load query feature----------")
        for i in tqdm(range(len(self.data))): 
            qid=self.data[i]['qid']
            if qid not in self.query_feat_dict:
                if self.debug:
                    self.query_feat_dict[qid] = flag_q
                else:
                    self.query_feat_dict[qid]=self._get_query_feat_by_qid(qid)
                    
    def _get_base_data(self,qid,vid,st,ed): 
        v_feature_all = self.video_feat_dict[vid]
        v_feature = v_feature_all[st:ed,:]
        q_feature = self.query_feat_dict[qid] 
        return q_feature, v_feature   
    
    def _loadPropLabel(self, filename):
        if os.path.exists(filename):
            prop_label_file = h5py.File(filename, 'r')
            self.cls_label=np.array(prop_label_file['cls_label'][:])
            self.reg_label=np.array(prop_label_file['reg_label'][:])
            prop_label_file.close()
            # self.action_frame_count = np.sum(self.cls_label.reshape((-1,self.cls_label.shape[-1])),axis=0)
            # self.action_frame_count=torch.Tensor(self.action_frame_count)
            return
        
        self.cls_label=[]
        self.reg_label=[]
        for i in tqdm(range(len(self.inputs))):
            cls_anc=[]
            reg_anc=[]
            vid = self.inputs[i][1]
            ed = self.inputs[i][3]
            relevant_windows = copy.deepcopy(self.data[self.inputs[i][5]]['relevant_windows'])
            for k in range(len(relevant_windows)):
                relevant_windows[k][0] /= 2
                relevant_windows[k][1] /= 2
                window_ed = relevant_windows[k][1]
                relevant_windows[k][1] = window_ed - relevant_windows[k][0]
                relevant_windows[k][0] = window_ed

            for j in range(0,len(self.anchor_windows)):
                v1 = np.zeros(2)
                v1[-1]=1
                v2 = np.zeros(2)
                v2[-1]=-1e3
                y_box = [ed, self.anchor_windows[j]]
                for target_window in relevant_windows:  
                    iou = self.calc_iou(y_box, target_window)
                    if iou >= self.pos_threshold or (j == len(self.anchor_windows)-1 and self.box_include(y_box, target_window)) or (j==0 and self.box_include(target_window, y_box)):
                        v1[0]=1
                        v1[-1]=0
                        v2[0]=1.0*(target_window[0]-y_box[0])/self.anchor_windows[j] #offset
                        v2[1]=np.log(1.0*max(1,target_window[1])/y_box[1]) #length
                cls_anc.append(v1)
                reg_anc.append(v2)
            cls_anc=np.stack(cls_anc, axis=0)
            reg_anc=np.stack(reg_anc, axis=0)
            self.cls_label.append(cls_anc)
            self.reg_label.append(reg_anc)

        self.cls_label=np.stack(self.cls_label,axis=0)
        self.reg_label=np.stack(self.reg_label,axis=0)

        os.makedirs('./output', exist_ok=True)
        outfile = h5py.File(filename, 'w')
        dset_cls = outfile.create_dataset('/cls_label', self.cls_label.shape, maxshape=self.cls_label.shape, chunks=True, dtype=np.float32)
        dset_cls[:,:] = self.cls_label[:,:]  
        dset_reg = outfile.create_dataset('/reg_label', self.reg_label.shape, maxshape=self.reg_label.shape, chunks=True, dtype=np.float32)
        dset_reg[:,:] = self.reg_label[:,:]  
        outfile.close()
        return
                                       
    def _makePropLabelUnit(self, i):
        video_name=self.inputs_all[i][0] # vid, st, ed, data_idx
        ed = self.inputs_all[i][2]
        
        cls_anc=[]
        reg_anc=[]
        for j in range(0,len(self.anchor_windows)):
            v1 = np.zeros(2)
            v1[-1]=1
            v2 = np.zeros(2)
            v2[-1]=-1e3
            y_box = [ed-1, self.anchor_windows[j]]
            
            subset_label=self._get_train_label_with_class(video_name,ed-self.anchor_windows[j],ed)
            idx_list = []
            for ii in range(0, subset_label.shape[0]):
                for jj in range(0, subset_label.shape[1]):
                    idx=int(subset_label[ii,jj])
                    if idx>0 and idx-1 not in idx_list:
                        idx_list.append(idx-1)
            
            for idx in idx_list:
                target_box = self.gt_action[video_name][idx]
                cls = int(target_box[2])
                iou = self.calc_iou(y_box,target_box)
                if iou >= self.pos_threshold or (j == len(self.anchors)-1 and self.box_include(y_box, target_box)) or (j==0 and self.box_include(target_box, y_box)):
                    v1[cls]=1
                    v1[-1]=0
                    v2[0]=1.0*(target_box[0]-y_box[0])/self.anchors[j]
                    v2[1]=np.log(1.0*max(1,target_box[1])/y_box[1])
            
            cls_anc.append(v1)
            reg_anc.append(v2)
        cls_anc=np.stack(cls_anc, axis=0)
        reg_anc=np.stack(reg_anc, axis=0)
        return cls_anc,reg_anc #cls_anc [K, C+1], reg_anc [K, 2]
    
    def calc_iou(self, a, b):
        st = a[0]-a[1]
        ed = a[0]
        target_st = b[0]-b[1]
        target_ed = b[0]
        sst = min(st, target_st)
        led = max(ed, target_ed)
        lst = max(st, target_st)
        sed = min(ed, target_ed)

        iou = (sed-lst) / max(led-sst,1)
        return iou

    def box_include(self, y, target): #is target is the larger box than y?
        st = y[0]-y[1]
        ed = y[0]
        target_st = target[0]-target[1]
        target_ed = target[0]
        
        detection_point = target_st #(target_st+target_ed)/2.0
        
        if ed > detection_point and target_st < st and target_ed > ed:
            return True
        return False    
                       
def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k == "cls_label":
            batched_data[k] = [dict(cls_label=e["model_inputs"]["cls_label"]) for e in batch]
            continue
        if k == "reg_label":
            batched_data[k] = [dict(reg_label=e["model_inputs"]["reg_label"]) for e in batch]
            continue
        if k == "query_feat":
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
        if k == "video_feat":
            batched_data[k] = pre_pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data

def prepare_batch_inputs(batched_model_inputs, device, opt=None, video_feat_pad=None, mask_pad=None, training=False, non_blocking=False):
    if not training:
        # 测试的时候，给video起始的左侧填零
        tmp = torch.zeros((batched_model_inputs["video_feat"][0].size(1)+opt.segment_size-1,batched_model_inputs["video_feat"][0].size(2)))
        tmp = [tmp, batched_model_inputs["video_feat"][0].squeeze(0)]
        tmp = pre_pad_sequences_1d(tmp, dtype=torch.float32, fixed_length=None)
        batched_model_inputs["video_feat"][0] = tmp[0][1].unsqueeze(0)
        batched_model_inputs["video_feat"][1] = tmp[1][1].unsqueeze(0)

    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    
    if "audio_feat" in batched_model_inputs:
        model_inputs["src_aud"] = batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking)
        model_inputs["src_aud_mask"] = batched_model_inputs["audio_feat"][1].to(device, non_blocking=non_blocking)

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    # only for TR-DETR
    if "pos_mask" in batched_model_inputs:
        targets['src_pos_mask']=batched_model_inputs["pos_mask"][0].to(device, non_blocking=non_blocking)

    if "cls_label" in batched_model_inputs:
        targets["cls_labels"] = [
            e["cls_label"].to(device, non_blocking=non_blocking)
            for e in batched_model_inputs["cls_label"]]
        targets["cls_labels"] = torch.stack(targets["cls_labels"],dim=0)
        if opt.minimal_unit == "window":
            targets["cls_labels"] = targets["cls_labels"].reshape(-1,targets["cls_labels"].size(2),targets["cls_labels"].size(3))
    if "reg_label" in batched_model_inputs:
        targets["reg_labels"] = [
            e["reg_label"].to(device, non_blocking=non_blocking)
            for e in batched_model_inputs["reg_label"]
            ]
        targets["reg_labels"] = torch.stack(targets["reg_labels"],dim=0)
        if opt.minimal_unit == "window":
            targets["reg_labels"] = targets["reg_labels"].reshape(-1,targets["reg_labels"].size(2),targets["reg_labels"].size(3))
    targets = None if len(targets) == 0 else targets

    return model_inputs, targets


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

