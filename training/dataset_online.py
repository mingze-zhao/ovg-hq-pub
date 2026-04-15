# import math
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
import h5py
import pickle
import json
from tqdm import tqdm
# from joblib import Parallel, delayed
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
                 max_windows=5, span_loss_type="l1", load_labels=True, segment_size=16, use_online=False, debug=False, anchor_windows=[], pos_threshold=None, label_file_path=None, opt=None, training=False):
        self.dset_name = dset_name
        self.domain = domain
        self.data_path = data_path
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_types = v_feat_types
        self.t_feat_type = opt.t_feat_type
        self.t_feat_dir = opt.t_feat_dir
        self.query_type = opt.query_type if "query_type" in opt else None
        self.opt = opt
        
        if max_v_l == -1:
            max_v_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.clip_len = clip_len

        # 新加
        self.data = self.load_data()
        self.qid_data_dict = {str(d['qid']): d for d in self.data}
        self.minimal_unit = opt.minimal_unit
        self.encoder_layer_type = opt.encoder_layer_type
        self.training = training
        self.anchor_windows=anchor_windows
        self.debug = debug
        self.pos_threshold = pos_threshold
        self.label_file_path = label_file_path
        self.eval_subset = opt.eval_subset
        self.clip_sub_sampling_rate = opt.clip_sub_sampling_rate
        self.query_type_list = ["image_r","image_c", "image_g", "segment_g"]


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
    
        self.segment_size = segment_size
        self.sequence_length = opt.sequence_length
        self._makeInputSeq() #返回整个Dataset的所有segment,[Nseg]
        self._get_video_query_feat_dict()
        if training:
            self._loadPropLabel(self.label_file_path.format("train"))
            if self.debug == False:
                self._filter_pos_inputs()
        else:
            self._loadPropLabel(self.label_file_path.format("eval"))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]   
        qid, vid, st, ed, data_idx, line_idx = input
        if self.dset_name == "tacos" and self.t_feat_type == "glove":
            if self.training:
                qid = str(line_idx)
            else:
                qid = str(line_idx+14226)

        if self.dset_name == "activitynet" and self.t_feat_type == "glove":
            if self.training:
                qid = str(line_idx)
            else:
                qid = str(line_idx+37421)

        if self.opt.dset_name == "qvhighlight_unify":
            if self.training:
                random.seed(self.opt.seed)
                rand_idx = random.randint(0,3)
                query_type = self.query_type_list[rand_idx]
            else:
                query_type = self.opt.eval_query_type
            if st >= 0:
                query_feat, video_feat = self._get_base_data(qid,vid,st,ed,query_type)
            else :
                query_feat, video_feat = self._get_base_data(qid,vid,0,ed,query_type)
        else:
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
                sal_label=torch.Tensor(self.sal_label[data_idx:data_idx+self.sequence_length])
            else:
                cls_label=torch.Tensor(self.cls_label[data_idx:data_idx+ed])
                reg_label=torch.Tensor(self.reg_label[data_idx:data_idx+ed])
                sal_label=torch.Tensor(self.sal_label[data_idx:data_idx+ed])
                # 下面这些视频label有问题，所以这里特殊处理一下
                if cls_label.shape[0] != video_feat.shape[0]:
                # if vid in ["KQyJtq52Jcw_660.0_810.0", "Z-L9RjWBTHg_360.0_510.0", "rNPSRSs3reQ_60.0_210.0", "IAbAn-MkMH8_510.0_660.0", "J4pIK7YehhQ_60.0_210.0","j7rJstUseKg_360.0_510.0","j7rJstUseKg_60.0_210.0","-Oc6gSWB_HA_60.0_210.0","G60-kHBEeZA_60.0_210.0","Ok-M_V_h-eY_210.0_360.0","Ok-M_V_h-eY_360.0_510.0","Ok-M_V_h-eY_60.0_210.0","S73Z-nM0GQE_60.0_210.0","S73Z-nM0GQE_210.0_360.0","S73Z-nM0GQE_360.0_510.0","S73Z-nM0GQE_660.0_810.0","S73Z-nM0GQE_510.0_660.0","6JnES9tDKy8_210.0_360.0","uSAGSbauHBs_510.0_660.0"]: # 这个视频的label有问题，所以这里特殊处理一下
                    cls_label=torch.Tensor(self.cls_label[data_idx:data_idx+ed-1])
                    reg_label=torch.Tensor(self.reg_label[data_idx:data_idx+ed-1])
                    sal_label=torch.Tensor(self.sal_label[data_idx:data_idx+ed-1])

        meta = self.data[line_idx]
        model_inputs = dict()
        if self.opt.dset_name == "qvhighlight_icq" and self.opt.query_type == "image_text":
            model_inputs["image_feat"] = query_feat["image_feat"]
            model_inputs["text_feat"] = query_feat["text_feat"]
        elif self.opt.dset_name == "qvhighlight_image" and self.opt.query_type == "image_text":
            model_inputs["image_feat"] = query_feat["image_feat"]
            model_inputs["text_feat"] = query_feat["text_feat"]
        elif self.opt.dset_name == "qvhighlight_segment" and self.opt.query_type == "segment_text":
            model_inputs["segment_feat"] = query_feat["segment_feat"]
            model_inputs["text_feat"] = query_feat["text_feat"]
        elif self.opt.dset_name == "qvhighlight_gen_segment" and self.opt.query_type == "segment_text":
            model_inputs["segment_feat"] = query_feat["segment_feat"]
            model_inputs["text_feat"] = query_feat["text_feat"]
        elif self.opt.dset_name == "qvhighlight_unify" and self.opt.query_type == "segment_text":
            model_inputs["teacher_segment_feat"] = query_feat["teacher_segment_feat"]
            model_inputs["teacher_text_feat"] = query_feat["teacher_text_feat"]
            model_inputs["segment_feat"] = query_feat["segment_feat"]
            model_inputs["text_feat"] = query_feat["text_feat"]
        else:    
            model_inputs["query_feat"] = query_feat
        model_inputs["video_feat"] = video_feat
        model_inputs["cls_label"] = cls_label
        model_inputs["reg_label"] = reg_label
        model_inputs["sal_label"] = sal_label
        model_inputs["input"] = input
        model_inputs["window_st_ed"] = [st,ed]

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

    def _makeInputSeq(self):
        data_idx=0
        self.inputs=[]
        if self.training:
            for index in range(0,len(self.data)):
                qid=self.data[index]['qid']
                vid=self.data[index]['vid']    
                duration = self.data[index]['duration']
                duration = round(duration / (self.clip_len * self.clip_sub_sampling_rate)) # 补充sub_sampling_rate逻辑，用来提高C3D feature 的训练测试速度
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
                            # if data_idx
                        else:
                            data_idx+=1
        else:
            for index in range(0,len(self.data)):
                qid=self.data[index]['qid']
                vid=self.data[index]['vid']    
                duration = self.data[index]['duration']
                duration = round(duration / (self.clip_len * self.clip_sub_sampling_rate))
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
        if self.dset_name == "tacos" and self.v_feat_types == "c3d":
            v_file_path = join(self.v_feat_dirs[0], "tall_c3d_features.hdf5")
            with h5py.File(v_file_path, 'r') as f:
                for video_name in tqdm(f):
                    self.video_feat_dict[video_name[:-4]] = self._get_video_feat_by_vid(video_name, f) # [:-4]: s35-d55.avi -> s35-d55
        elif self.dset_name == "activitynet" and self.v_feat_types == "c3d":
            subset = "train" if self.training else self.eval_subset
            v_file_path = join(self.v_feat_dirs[0], "activitynet_v1-3_c3d_{}.hdf5".format(subset))
            with h5py.File(v_file_path, 'r') as f:
                for video_name in tqdm(f):
                    self.video_feat_dict[video_name] = self._get_video_feat_by_vid(video_name, f) # [:-4]: s35-d55.avi -> s35-d55

        else: 
            for i in tqdm(range(len(self.data))): 
                video_name=self.data[i]['vid']
                if video_name not in self.video_feat_dict:
                    if self.debug:
                        self.video_feat_dict[video_name] = flag_v
                    else:
                        self.video_feat_dict[video_name]=self._get_video_feat_by_vid(video_name)
                
        print("--------load query feature----------")
        if self.dset_name in ["tacos", "activitynet"] and self.t_feat_type == "glove":
            subset = "train" if self.training else self.eval_subset
            t_file_path = join(self.t_feat_dir, "bert-base-uncased_language_tokens_features_{}.hdf5".format(subset))
            with h5py.File(t_file_path, 'r') as f:
                for qid in tqdm(f):
                    self.query_feat_dict[qid] = self._get_query_feat_by_qid(qid, f)

        elif self.dset_name == "qvhighlight_segment":
            for i in tqdm(range(len(self.data))):
                qid=self.data[i]['qid']
                vid=self.data[i]['vid']
                segment=self.data[i]['segment_query']
                if qid not in self.query_feat_dict:
                    if self.debug:
                        self.query_feat_dict[qid] = flag_q
                    else:
                        self.query_feat_dict[qid]=self._get_query_feat_by_qid(qid, vid, segment)

        elif self.dset_name == "qvhighlight_unify":
            for i in tqdm(range(len(self.data))):
                qid=self.data[i]['qid']
                vid=self.data[i]['vid']
                # segment=self.data[i]['segment_query']
                for query_type in ["image_r","image_c","image_g","segment_g"]:
                    if qid not in self.query_feat_dict:
                        if self.debug:
                            self.query_feat_dict[qid] = flag_q
                        else:
                            if query_type not in self.query_feat_dict:
                                self.query_feat_dict[query_type] = {}
                            self.query_feat_dict[query_type].update({qid: self._get_query_feat_by_qid(qid, vid, query_type=query_type)})

        else:
            for i in tqdm(range(len(self.data))): 
                qid=self.data[i]['qid']
                if qid not in self.query_feat_dict:
                    # if self.debug:
                        # self.query_feat_dict[qid] = flag_q
                    # else:
                    self.query_feat_dict[qid]=self._get_query_feat_by_qid(qid)
                   
    def _get_query_feat_by_qid(self, qid, vid=None, segment=None, query_type=None, f = None):
        if f is not None:
            _feat = f[qid][:]
            _feat = l2_normalize_np_array(_feat)
            return torch.from_numpy(_feat)

        if self.dset_name == 'tvsum' or self.dset_name == 'youtube_highlight':
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)
            return torch.from_numpy(q_feat['token']) if self.dset_name == 'tvsum' else torch.from_numpy(q_feat['last_hidden_state'])
        elif self.dset_name == 'qvhighlight_icq':
            if self.query_type == "image":
                img_feat_path = join(self.q_feat_dir["image"], f"qid{qid}.npz")
                img_feat = np.load(img_feat_path)['features'].astype(np.float32)
                img_feat = l2_normalize_np_array(img_feat)
                return torch.from_numpy(img_feat).unsqueeze(0)
            elif self.query_type == "text":
                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid.split('_')[0]}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                return torch.from_numpy(text_feat)
            elif self.query_type == "image_text":
                img_feat_path = join(self.q_feat_dir["image"], f"qid{qid}.npz")
                img_feat = np.load(img_feat_path)['features'].astype(np.float32)
                img_feat = l2_normalize_np_array(img_feat)
                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid.split('_')[0]}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                q_feat = {"image_feat":torch.from_numpy(img_feat).unsqueeze(0), "text_feat":torch.from_numpy(text_feat)}
                return q_feat
            
        elif self.dset_name == 'qvhighlight_image':
            if self.query_type == "image":
                img_feat_path = join(self.q_feat_dir["image"], f"qid{qid}.npz")
                img_feat = np.load(img_feat_path)['features'].astype(np.float32)
                img_feat = l2_normalize_np_array(img_feat)
                return torch.from_numpy(img_feat).unsqueeze(0)
            elif self.query_type == "text":
                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid.split('_')[0]}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                return torch.from_numpy(text_feat)
            elif self.query_type == "image_text":
                img_feat_path = join(self.q_feat_dir["image"], f"qid{qid}.npz")
                img_feat = np.load(img_feat_path)['features'].astype(np.float32)
                img_feat = l2_normalize_np_array(img_feat)
                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                q_feat = {"image_feat":torch.from_numpy(img_feat).unsqueeze(0), "text_feat":torch.from_numpy(text_feat)}
                return q_feat

        elif self.dset_name == 'qvhighlight_segment':
            v_feat_list = []
            if self.query_type == "segment":
                v_feat = self._get_video_feat_by_vid(vid)
                st, ed = segment[0]//2, segment[1]//2
                v_feat = v_feat[st:ed]
                return v_feat
            elif self.query_type == "text":
                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                return torch.from_numpy(text_feat)
            elif self.query_type == "segment_text":
                v_feat = self._get_video_feat_by_vid(vid)
                st, ed = segment[0]//2, segment[1]//2
                v_feat = v_feat[st:ed]

                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                q_feat = {"segment_feat":v_feat, "text_feat":torch.from_numpy(text_feat)}
                return q_feat

        elif self.dset_name == 'qvhighlight_gen_segment':
            v_feat_list = []
            if self.query_type == "segment":
                v_feat = self._get_video_feat_by_qid(qid)
                return v_feat
            elif self.query_type == "text":
                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                return torch.from_numpy(text_feat)
            elif self.query_type == "segment_text":
                v_feat = self._get_video_feat_by_qid(qid)
                text_feat_path = join(self.q_feat_dir["text"], f"qid{qid}.npz")
                text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    text_feat = text_feat[:self.max_q_l]
                text_feat = l2_normalize_np_array(text_feat)
                q_feat = {"segment_feat":v_feat, "text_feat":torch.from_numpy(text_feat)}
                return q_feat

        elif self.dset_name == 'qvhighlight_unify':
            if query_type in ["image_r", "image_c", "image_g", "segment_g"]:
                v_feat = self._get_video_feat_by_qid(qid, query_type)
            # elif query_type in ["ori_segment"]:
            #     v_feat = self._get_video_feat_by_vid(vid)
            #     st, ed = segment[0]//2, segment[1]//2
            #     v_feat = v_feat[st:ed]

            # teacher_v_feat = self._get_video_feat_by_vid(vid)
            # st, ed = segment[0]//2, segment[1]//2
            # teacher_v_feat = teacher_v_feat[st:ed]

            teacher_v_feat = self._get_video_feat_by_qid(qid, query_type="segment_g")
            teachaer_t_feat_dir = self.q_feat_dir["text"] 
            teachaer_t_feat_path = join(teachaer_t_feat_dir, f"qid{qid}.npz")
            teachaer_t_feat = np.load(teachaer_t_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                teachaer_t_feat = teachaer_t_feat[:self.max_q_l]
            teachaer_t_feat = l2_normalize_np_array(teachaer_t_feat)

            text_feat_dir = self.q_feat_dir["text_c"] if query_type == "image_c" else self.q_feat_dir["text"] 
            text_feat_path = join(text_feat_dir, f"qid{qid}.npz")

            text_feat = np.load(text_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                text_feat = text_feat[:self.max_q_l]
            text_feat = l2_normalize_np_array(text_feat)
            q_feat = {"teacher_segment_feat": teacher_v_feat, "teacher_text_feat": torch.from_numpy(teachaer_t_feat), "segment_feat":v_feat, "text_feat":torch.from_numpy(text_feat)}
            return q_feat
        
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

    def _get_video_feat_by_vid(self, vid, feat_file=None, query_type=None):
        v_feat_list = []
        if feat_file is not None:
            _feat = feat_file[vid][:]
            _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        else:
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
    
    def _get_video_feat_by_qid(self, qid, query_type=None):
        v_feat_list = []
        if query_type is None:
            for _feat_dir in self.t_feat_dir["segment"]:
                _feat_path = join(_feat_dir, f"qid{qid}.npz")
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        else:
            for _feat_dir in self.t_feat_dir["segment"][query_type]:
                _feat_path = join(_feat_dir, f"qid{qid}.npz")
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)
 
    def _get_base_data(self,qid,vid,st,ed,query_type=None): 
        v_feature_all = self.video_feat_dict[vid]
        v_feature = v_feature_all[st:ed,:]
        if query_type is None:
            q_feature = self.query_feat_dict[qid] 
        else:
            q_feature = self.query_feat_dict[query_type][qid] 
        return q_feature, v_feature   
    
    def _loadPropLabel(self, filename):
        if os.path.exists(filename):
            prop_label_file = h5py.File(filename, 'r')
            self.cls_label=np.array(prop_label_file['cls_label'][:])
            self.reg_label=np.array(prop_label_file['reg_label'][:])
            self.sal_label=np.array(prop_label_file['sal_label'][:])
            prop_label_file.close()
            # self.action_frame_count = np.sum(self.cls_label.reshape((-1,self.cls_label.shape[-1])),axis=0)
            # self.action_frame_count=torch.Tensor(self.action_frame_count)
            return
        
        self.cls_label=[]
        self.reg_label=[]
        self.sal_label=[]
        for i in tqdm(range(len(self.inputs))): # 遍历所有InputSeq
            cls_anc=[]
            reg_anc=[]
            sal_idx=[]
            vid = self.inputs[i][1]
            st = 0 if self.inputs[i][2] < 0 else self.inputs[i][2]
            ed = self.inputs[i][3]
            relevant_windows = copy.deepcopy(self.data[self.inputs[i][5]]['relevant_windows']) # 获取对应的relevant_windows
            scores = np.array(self.data[self.inputs[i][5]]["saliency_scores"])
            rel_clip_ids = self.data[self.inputs[i][5]]["relevant_clip_ids"] 

            # Moment格式: (st, ed) -> (ed, length) 
            for k in range(len(relevant_windows)): # GT格式: (st, ed) -> (ed, length) (4.5, 7.3) ->(7, 2)
                relevant_windows[k][0] = round(relevant_windows[k][0] / (self.clip_len * self.clip_sub_sampling_rate))
                relevant_windows[k][1] = round(relevant_windows[k][1] / (self.clip_len * self.clip_sub_sampling_rate))
                window_st = relevant_windows[k][0]
                window_ed = relevant_windows[k][1]
                relevant_windows[k][1] = window_ed - window_st
                relevant_windows[k][0] = window_ed

            # moment retrieval label
            for j in range(0,len(self.anchor_windows)): # 遍历所有anchor, 每个anchor与GT计算IOU，进而分配标签
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
            
            # highlight detection label
            min_idx = -1
            max_idx = -1
            if set(rel_clip_ids).intersection(set(range(st,ed))):
                intersection_idx = list(set(rel_clip_ids).intersection(set(range(st,ed))))
                score_array_intersection_idx = [idx - st for idx in intersection_idx]
                score_array = np.zeros(ed - st)
                
                agg_scores = np.sum(scores, 1)
                agg_scores = agg_scores[:len(intersection_idx)]

                score_array[score_array_intersection_idx] = agg_scores

                min_idx = np.where(score_array == np.min(score_array))[0]
                min_idx = np.random.choice(min_idx)
                max_idx = np.where(score_array == np.max(score_array))[0]
                max_idx = np.random.choice(max_idx)
            sal_idx = [min_idx, max_idx]

            cls_anc=np.stack(cls_anc, axis=0)
            reg_anc=np.stack(reg_anc, axis=0)
            sal_idx=np.stack(sal_idx, axis=0)
            self.cls_label.append(cls_anc)
            self.reg_label.append(reg_anc)
            self.sal_label.append(sal_idx)
        self.cls_label=np.stack(self.cls_label,axis=0)
        self.reg_label=np.stack(self.reg_label,axis=0)
        self.sal_label=np.stack(self.sal_label,axis=0)

        os.makedirs('./output', exist_ok=True)
        outfile = h5py.File(filename, 'w')
        dset_cls = outfile.create_dataset('/cls_label', self.cls_label.shape, maxshape=self.cls_label.shape, chunks=True, dtype=np.float32)
        dset_cls[:,:] = self.cls_label[:,:]  
        dset_reg = outfile.create_dataset('/reg_label', self.reg_label.shape, maxshape=self.reg_label.shape, chunks=True, dtype=np.float32)
        dset_reg[:,:] = self.reg_label[:,:]  
        dset_sal = outfile.create_dataset('/sal_label', self.sal_label.shape, maxshape=self.sal_label.shape, chunks=True, dtype=np.float32)
        dset_sal[:,:] = self.sal_label[:,:]  
        outfile.close()
        return                  
    
    def _filter_pos_inputs(self):         
        inputs_flag = []
        for i, input in tqdm(enumerate(self.inputs)):     
            data_idx = input[4]
            cls_label=torch.Tensor(self.cls_label[data_idx:data_idx+self.sequence_length])
            if torch.sum(cls_label[:,:,0]) != 0:
                inputs_flag.append(input)
        self.inputs = inputs_flag
    
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

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        return datalist

# def start_end_collate(batch):
#     # return
#     batch_meta = [e["meta"] for e in batch]
#     model_inputs = [e["model_inputs"] for e in batch]
#     batched_data = {}
#     for k in model_inputs[0].keys():
#         if k == "cls_label" or k == "reg_label" or k == "sal_label":
#             batched_data[k] = [{k: e[k]} for e in model_inputs]
#         elif k == "input":
#             batched_data[k] = [dict(input=e["input"]) for e in model_inputs]
#         elif k == "window_st_ed":
#             start_end_pairs = [e["window_st_ed"] for e in model_inputs]
#             split_data = [
#                 [s, s + 16] 
#                 for start, end in start_end_pairs
#                 for s in range(start, end - 15)
#             ]
#             batched_data[k] = torch.tensor(split_data, dtype=torch.int64)
#         elif k in {"query_feat", "video_feat", "image_feat", "text_feat", "segment_feat"}:
#             pad_func = pre_pad_sequences_1d if k == "video_feat" else pad_sequences_1d
#             batched_data[k] = pad_func(
#                 [e[k] for e in model_inputs], dtype=torch.float32
#             )
#     return batch_meta, batched_data

def start_end_collate(batch):
    # return
    batch_meta = [e["meta"] for e in batch]
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "cls_label":
            batched_data[k] = [dict(cls_label=e["model_inputs"]["cls_label"]) for e in batch]
        elif k == "input":
            batched_data[k] = [dict(input=e["model_inputs"]["input"]) for e in batch]
        elif k == "window_st_ed":
            batched_data[k] = [e["model_inputs"]["window_st_ed"] for e in batch]
            split_data = []
            for i in range(len(batched_data[k])):
                start = batched_data[k][i][0]  # 起始值
                end = batched_data[k][i][1]  # 终止值
                segment_length = 16        # 每个片段长度为16
                # 计算可滑动窗口的起始位置
                num_segments = end - start - segment_length + 1
                if num_segments > 0:
                    # 生成每个片段的起始值
                    starts = np.arange(start, end - segment_length + 1)
                    # 创建新的样本，使用列表推导
                    for s in starts:
                        split_data.append([s, s + segment_length])
            batched_data[k] = torch.tensor(split_data)
        elif k == "reg_label":
            batched_data[k] = [dict(reg_label=e["model_inputs"]["reg_label"]) for e in batch]
        elif k == "sal_label":
            batched_data[k] = [dict(sal_label=e["model_inputs"]["sal_label"]) for e in batch]
        elif k == "query_feat":
            batched_data[k] = list(pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None))
        elif k == "video_feat":
            batched_data[k] = list(pre_pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None))
        elif k == "image_feat":
            batched_data[k] = list(pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None))
        elif k == "text_feat":
            batched_data[k] = list(pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None))
        elif k == "segment_feat":
            batched_data[k] = list(pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None))
        elif k == "teacher_segment_feat":
            batched_data[k] = list(pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None))
        elif k == "teacher_text_feat":
            batched_data[k] = list(pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None))

    return batch_meta, batched_data

def prepare_batch_inputs(batched_model_inputs, device, opt=None, video_feat_pad=None, mask_pad=None, training=False, non_blocking=False):
    torch.set_num_threads(1)
    if not training:
        # 测试的时候，给video起始的左侧填零
        tmp = torch.zeros((batched_model_inputs["video_feat"][0].size(1)+opt.segment_size-1,batched_model_inputs["video_feat"][0].size(2)))
        tmp = [tmp, batched_model_inputs["video_feat"][0].squeeze(0)]
        tmp = pre_pad_sequences_1d(tmp, dtype=torch.float32, fixed_length=None)
        batched_model_inputs["video_feat"][0] = tmp[0][1].unsqueeze(0)
        batched_model_inputs["video_feat"][1] = tmp[1][1].unsqueeze(0)

    if opt.dset_name in ["qvhighlight_icq", "qvhighlight_image"] and opt.query_type == "image_text":
        model_inputs = dict(
            src_img=batched_model_inputs["image_feat"][0].to(device, non_blocking=non_blocking),
            src_img_mask=batched_model_inputs["image_feat"][1].to(device, non_blocking=non_blocking),
            src_txt=batched_model_inputs["text_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["text_feat"][1].to(device, non_blocking=non_blocking),
            src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
            window_st_ed = batched_model_inputs["window_st_ed"].to(device, non_blocking=non_blocking),
            input = batched_model_inputs["input"],

        ) 
    elif opt.dset_name in ["qvhighlight_segment", "qvhighlight_gen_segment"] and opt.query_type == "segment_text":
        model_inputs = dict(
            src_sgm=batched_model_inputs["segment_feat"][0].to(device, non_blocking=non_blocking),
            src_sgm_mask=batched_model_inputs["segment_feat"][1].to(device, non_blocking=non_blocking),
            src_txt=batched_model_inputs["text_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["text_feat"][1].to(device, non_blocking=non_blocking),
            src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
            window_st_ed = batched_model_inputs["window_st_ed"].to(device, non_blocking=non_blocking),
            input = batched_model_inputs["input"],
        )
    elif opt.dset_name in ["qvhighlight_unify"] and opt.query_type == "segment_text":
        model_inputs = dict(
            src_sgm=batched_model_inputs["segment_feat"][0].to(device, non_blocking=non_blocking),
            src_sgm_mask=batched_model_inputs["segment_feat"][1].to(device, non_blocking=non_blocking),
            src_sgm_teacher=batched_model_inputs["teacher_segment_feat"][0].to(device, non_blocking=non_blocking),
            src_sgm_mask_teacher=batched_model_inputs["teacher_segment_feat"][1].to(device, non_blocking=non_blocking),
            src_txt_teacher=batched_model_inputs["teacher_text_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask_teacher=batched_model_inputs["teacher_text_feat"][1].to(device, non_blocking=non_blocking),
            src_txt=batched_model_inputs["text_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["text_feat"][1].to(device, non_blocking=non_blocking),
            src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
            window_st_ed = batched_model_inputs["window_st_ed"].to(device, non_blocking=non_blocking),
            input = batched_model_inputs["input"],
        )
    else:
        model_inputs = dict(
            src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
            src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
            window_st_ed = batched_model_inputs["window_st_ed"].to(device, non_blocking=non_blocking),
             input = batched_model_inputs["input"],
        )
    
    targets = {}
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
    if "sal_label" in batched_model_inputs:
        targets["sal_labels"] = [
            e["sal_label"].to(device, non_blocking=non_blocking)
            for e in batched_model_inputs["sal_label"]
            ]
        targets["sal_labels"] = torch.stack(targets["sal_labels"],dim=0)
        if opt.minimal_unit == "window":
            targets["sal_labels"] = targets["sal_labels"].reshape(-1,targets["reg_labels"].size(2))
    targets = None if len(targets) == 0 else targets

    return model_inputs, targets


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_relevant_video_feat(batch_meta, video_feat_dict):
    relevant_video_feat = {}
    for i in range(len(batch_meta)):
        vid = batch_meta[i]['vid']
        if vid not in relevant_video_feat.keys():
            new_video_feat = {vid:video_feat_dict[vid].cuda()}
            relevant_video_feat.update(new_video_feat)
    return relevant_video_feat