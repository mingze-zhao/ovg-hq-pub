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

import os
import json
import pickle
import time
import torch
from collections import OrderedDict, Counter
import pandas as pd
import numpy as np
import zipfile
from glob import glob
from urllib.parse import quote
import sys
import socket
from lighthouse.common.online_suppress_net import SuppressNet
from tqdm import tqdm
import csv

def metricstocsv(epoch, json_data, csv_file_path):
    header_order = [
        "Epoch",
        "OFF-R1@0.5","On-Avg-R1@0.5_pos","On-Avg-R1@0.5_pos_gamma","On-Avg-R1@0.5_minus",
        "OFF-R1@0.7","On-Avg-R1@0.7_pos","On-Avg-R1@0.7_pos_gamma","On-Avg-R1@0.7_minus",
        "OFF-mAP@0.5","On-Avg-mAP@0.5_pos","On-Avg-mAP@0.5_pos_gamma","On-Avg-mAP@0.5_minus",
        "OFF-mAP@0.75","On-Avg-mAP@0.75_pos","On-Avg-mAP@0.75_pos_gamma","On-Avg-mAP@0.75_minus",
        "OFF-mAP@Avg","On-Avg-mAP@Avg_pos","On-Avg-mAP@Avg_pos_gamma","On-Avg-mAP@Avg_minus",
        "On3-R1@0.5_pos","On3-R1@0.5_pos_gamma","On3-R1@0.5_minus",
        "On3-R1@0.7_pos","On3-R1@0.7_pos_gamma","On3-R1@0.7_minus",
        "On3-mAP@0.5_pos","On3-mAP@0.5_pos_gamma","On3-mAP@0.5_minus",
        "On3-mAP@0.75_pos","On3-mAP@0.75_pos_gamma","On3-mAP@0.75_minus",
        "On3-mAP@Avg_pos","On3-mAP@Avg_pos_gamma","On3-mAP@Avg_minus",
        "HL-min-VeryGood-mAP","HL-min-VeryGood-Hit1"
        ]
    # 将数据提取到一个按顺序排列的列表
    data = {key: json_data["brief"].get(key, None) for key in header_order[1:]}  # 排除第一个元素 "Epoch"
    
    # 检查文件是否存在且非空
    file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0
    
    # 写入 CSV 文件
    with open(csv_file_path, mode='a', newline='') as file:  # 使用 'a' 模式追加写入
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header_order)  # 写入表头
        # 写入数据行，添加 epoch
        writer.writerow([epoch] + [data[key] for key in header_order[1:]])  # 在数据前加上 epoch

    print(f"数据已写入 {csv_file_path}")

def write_log(opt, epoch_i, loss_meters, metrics=None, mode='train', iter_modal=''):
    # log
    if mode == 'train':
        if "loss_val" in opt:
            to_write = opt.train_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i+1,
                loss_str=" ".join(["{} {:.4f}".format(k, v.val) for k, v in loss_meters.items()]))
        else:
            to_write = opt.train_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i+1,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        filename = opt.train_log_filepath
    else:
        if "loss_val" in opt:
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.val) for k, v in loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics))
        else:
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics))
        filename = f"val{iter_modal}.log"
    
    with open(filename, "a") as f:
        f.write(to_write)

def save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch_i,
        "opt": opt
    }
    if not os.path.exists(opt.ckpt_filepath):
        os.makedirs(opt.ckpt_filepath)
    torch.save(checkpoint, os.path.join(opt.ckpt_filepath, f"epoch{epoch_i}.ckpt"))

def rename_latest_to_best(latest_file_paths):
    best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
    for src, tgt in zip(latest_file_paths, best_file_paths):
        os.renames(src, tgt)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def convert_to_seconds(hms_time):
    """ convert '00:01:12' to 72 seconds.
    :hms_time (str): time in comma separated string, e.g. '00:01:12'
    :return (int): time in seconds, e.g. 72
    """
    times = [float(t) for t in hms_time.split(":")]
    return times[0] * 3600 + times[1] * 60 + times[2]


def get_video_name_from_url(url):
    return url.split("/")[-1][:-4]


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_dirs=None, exclude_extensions=None,
                 exclude_dirs_substring=None):
    """make a zip file of root_dir, save it to save_path.
    exclude_paths will be excluded if it is a subdir of root_dir.
    An enclosing_dir is added is specified.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for d in subdirs:
                    if exclude_dirs_substring in d:
                        to_rm.append(d)
                for e in to_rm:
                    subdirs.remove(e)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1, part=False):
        if part:
            if val.nelement() == 0:
                return
            else:
                self.max = max(val.max(), self.max)
                self.min = min(val.min(), self.min)
                self.val = val.mean()
                self.sum += val.sum()
                self.count += val.nelement()
                self.avg = self.sum / self.count
        else:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def dissect_by_lengths(np_array, lengths, dim=0, assert_equal=True):
    """Dissect an array (N, D) into a list a sub-array,
    np_array.shape[0] == sum(lengths), Output is a list of nd arrays, singlton dimention is kept"""
    if assert_equal:
        assert len(np_array) == sum(lengths)
    length_indices = [0, ]
    for i in range(len(lengths)):
        length_indices.append(length_indices[i] + lengths[i])
    if dim == 0:
        array_list = [np_array[length_indices[i]:length_indices[i+1]] for i in range(len(lengths))]
    elif dim == 1:
        array_list = [np_array[:, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    elif dim == 2:
        array_list = [np_array[:, :, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    else:
        raise NotImplementedError
    return array_list


def get_ratio_from_counter(counter_obj, threshold=200):
    keys = counter_obj.keys()
    values = counter_obj.values()
    filtered_values = [counter_obj[k] for k in keys if k > threshold]
    return float(sum(filtered_values)) / sum(values)


def get_counter_dist(counter_object, sort_type="none"):
    _sum = sum(counter_object.values())
    dist = {k: float(f"{100 * v / _sum:.2f}") for k, v in counter_object.items()}
    if sort_type == "value":
        dist = OrderedDict(sorted(dist.items(), reverse=True))
    return dist


def get_show_name(vid_name):
    """
    get tvshow name from vid_name
    :param vid_name: video clip name
    :return: tvshow name
    """
    show_list = ["friends", "met", "castle", "house", "grey"]
    vid_name_prefix = vid_name.split("_")[0]
    show_name = vid_name_prefix if vid_name_prefix in show_list else "bbt"
    return show_name


def get_abspaths_by_ext(dir_path, ext=(".jpg",)):
    """Get absolute paths to files in dir_path with extensions specified by ext.
    Note this function does work recursively.
    """
    if isinstance(ext, list):
        ext = tuple(ext)
    if isinstance(ext, str):
        ext = tuple([ext, ])
    filepaths = [os.path.join(root, name)
                 for root, dirs, files in os.walk(dir_path)
                 for name in files
                 if name.endswith(tuple(ext))]
    return filepaths


def get_basename_no_ext(path):
    """ '/data/movienet/240p_keyframe_feats/tt7672188.npz' --> 'tt7672188' """
    return os.path.splitext(os.path.split(path)[1])[0]


def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

def save_sh_n_codes(work_dir, ignore_dir=['exp', 'annotation', 'data', 'label_output', 'visualize', 'wandb',]):
    os.makedirs(work_dir, exist_ok=True)
    name = os.path.join(work_dir, 'code.zip')
    name = os.listdir(work_dir)
    for i in name:
        path = f"{work_dir}/{i}"
        if "run_gpu" in path:
            os.remove(path)

    name = os.path.join(work_dir, 'run_{}.sh'.format(socket.gethostname()))
    with open(name, 'w') as f:
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')

    name = os.path.join(work_dir, 'code.zip')
    with zipfile.ZipFile(name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:

        first_list = glob('.vscode', recursive=True) + glob('*', recursive=True)
        first_list = [i for i in first_list if i not in ignore_dir]

        file_list = []
        patterns = [x + '/**' for x in first_list]
        for pattern in patterns:
            file_list.extend(glob(pattern, recursive=True))

        file_list = [x[:-1] if x[-1] == "/" else x for x in file_list]
        for filename in file_list:
            zf.write(filename)
    print("code saved")
    


def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    result_dict_r1_ap={}
    result_dict_r1_online={}
    proposal_dict=[]
    
    anchors=opt['anchor_windows']
                                            
    for i in tqdm(range(len(dataset.data))):
        qid = str(dataset.data[i]['qid'])
        frame_to_time = 100.0* (opt.clip_length * opt.clip_sub_sampling_rate)
        
        for idx in range(0,int(dataset.data[i]['duration'] / (opt.clip_length * opt.clip_sub_sampling_rate))):
            if idx >= len(output_cls[qid]):
                break
            cls_anc = output_cls[qid][idx]
            reg_anc = output_reg[qid][idx]
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                if cls_anc[anc_idx][0]<opt['threshold']:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                if st<0:
                    st=0
                if ed>int(dataset.data[i]['duration'] / (opt.clip_length * opt.clip_sub_sampling_rate)):
                    ed=int(dataset.data[i]['duration'] / (opt.clip_length * opt.clip_sub_sampling_rate))
                tmp_dict={}
                tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                tmp_dict["score"]= cls_anc[anc_idx][0]*1.0
                tmp_dict["gentime"]= (idx+1)*frame_to_time/100.0
                proposal_anc_dict.append(tmp_dict)

            if opt.per_frame_nms:
                proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=0.3) 
            proposal_dict+=proposal_anc_dict
        proposal_dict = non_max_suppression(proposal_dict, overlapThresh=0.1)
        # proposal_dict = sorted(proposal_dict, key=lambda proposal:proposal['score'], reverse=True)
        
        # if proposal_dict is [], add a proposal with score 0.0 to simulate no prediction for evaluation
        if proposal_dict == []:
            proposal_dict.append({'segment':[0.0, 0.0], 'score':0.0, 'gentime':0.0})
                            
        result_dict_r1_online[qid]=proposal_dict
        result_dict_r1_ap[qid]=sorted(proposal_dict, key=lambda proposal:proposal['score'], reverse=True)
        proposal_dict=[]
        
    return result_dict_r1_ap, result_dict_r1_online
    
def non_max_suppression(proposals, overlapThresh=0.3):
    # if there are no intervals, return an empty list
    if len(proposals) == 0:
        return []

    # initialize the list of picked indexes
    pick = []
    
    sorted_proposal = sorted(proposals, key=lambda proposal:proposal['score'], reverse=True)
    idx=0
    total_proposal= len(sorted_proposal)
    while idx < total_proposal: 
        proposal = sorted_proposal[idx]
        st = proposal['segment'][0]
        ed = proposal['segment'][1]
        
        delete_item = []
        for j in range(idx+1, total_proposal):
            target_proposal = sorted_proposal[j]
            target_st = target_proposal['segment'][0]
            target_ed = target_proposal['segment'][1]
            
            sst = np.minimum(st, target_st)
            led = np.maximum(ed, target_ed)
            lst = np.maximum(st, target_st)
            sed = np.minimum(ed, target_ed)
            
            iou = (sed-lst) / max(led-sst,1)
            if(iou > overlapThresh):
                delete_item.append(target_proposal)
                    
        for item in delete_item:
            sorted_proposal.remove(item)
        total_proposal=len(sorted_proposal)
        idx+=1
        
    return sorted_proposal
    
def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["sup_checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    result_dict={}
    proposal_dict=[]
    
    unit_size = opt['segment_size']
    anchors=opt['anchor_windows']
                                             
    for i in tqdm(range(len(dataset.data))):
        qid = str(dataset.data[i]['qid'])
        if dataset.dset_name == "qvhighlight":
            frame_to_time = 100.0* 2
        conf_queue = torch.zeros((unit_size,1)) 
        
        for idx in range(0,int(dataset.data[i]['duration'] / (opt.clip_length * opt.clip_sub_sampling_rate))):
            cls_anc = output_cls[qid][idx]
            reg_anc = output_reg[qid][idx]
                    
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                if cls_anc[anc_idx][0]<opt['threshold']:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                if st<0:
                    st=0
                if ed>int(dataset.data[i]['duration'] / (opt.clip_length * opt.clip_sub_sampling_rate)):
                    ed=int(dataset.data[i]['duration'] / (opt.clip_length * opt.clip_sub_sampling_rate))
                tmp_dict={}
                tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                tmp_dict["score"]= cls_anc[anc_idx][0]*1.0
                tmp_dict["gentime"]= idx*frame_to_time/100.0
                proposal_anc_dict.append(tmp_dict)
                          
            if opt.per_frame_nms:
                proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=0.3) 

            conf_queue[:-1,0]=conf_queue[1:,0].clone()
            conf_queue[-1,0]=0
            for proposal in proposal_anc_dict:
                conf_queue[-1,0]=proposal["score"]
            
            minput = conf_queue.unsqueeze(0)
            suppress_conf = model(minput.cuda())
            suppress_conf=suppress_conf.squeeze(0).detach().cpu().numpy()
            
            if suppress_conf[0] > opt['sup_threshold']:
                for proposal in proposal_anc_dict:
                    # if check_overlap_proposal(proposal_dict, proposal, overlapThresh=0.3) is None:
                    proposal_dict.append(proposal)

        proposal_dict = sorted(proposal_dict, key=lambda proposal:proposal['score'], reverse=True)
        result_dict[qid]=proposal_dict
        proposal_dict=[]
        
    return result_dict

def check_overlap_proposal(proposal_list, new_proposal, overlapThresh=0.3):
    for proposal in proposal_list:
        st = proposal['segment'][0]
        ed = proposal['segment'][1]
        
        new_st = new_proposal['segment'][0]
        new_ed = new_proposal['segment'][1]
        
        sst = np.minimum(st, new_st)
        led = np.maximum(ed, new_ed)
        lst = np.maximum(st, new_st)
        sed = np.minimum(ed, new_ed)
        
        iou = (sed-lst) / max(led-sst,1)
        if(iou > overlapThresh):
            return proposal
    return None
