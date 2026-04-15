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
os.environ["WANDB_MODE"]="offline"
import sys
import time
import json
import pprint
import random
import argparse
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from collections import defaultdict
from glob import glob

pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from easydict import EasyDict

from training.config import BaseOptions
from training.dataset_online import StartEndDataset, start_end_collate, prepare_batch_inputs, get_relevant_video_feat
# from training.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from training.cg_detr_dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs
from training.evaluate import eval_epoch, start_inference, setup_model

from lighthouse.common.utils.basic_utils import AverageMeter, dict_to_markdown, write_log, save_checkpoint, rename_latest_to_best, save_sh_n_codes, metricstocsv
from lighthouse.common.utils.model_utils import count_parameters, ModelEMA

from lighthouse.common.loss_func import VTCLoss
from lighthouse.common.loss_func import CTC_Loss
import wandb

import logging
import sys
import shutil
from pyinstrument import Profiler
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def allocate_full_gpu_memory():
    """
    在训练开始时占据当前GPU所有可用显存，避免动态分配
    """
    if torch.cuda.is_available():
        logger.info("正在占据当前GPU显存...")
        
        # 只使用当前GPU
        device_id = torch.cuda.current_device()
        device = torch.device(f'cuda:{device_id}')
        
        # 获取GPU总显存（字节）
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        # 预留一些显存给系统（约5%）
        allocate_memory = int(total_memory * 0.95)
        
        try:
            # 创建一个大tensor来占据显存，然后立即释放
            # 这会强制PyTorch的缓存分配器预留这部分显存
            dummy_tensor = torch.empty(allocate_memory // 4, dtype=torch.float32, device=device)
            del dummy_tensor
            torch.cuda.empty_cache()
            
            logger.info(f"GPU {device_id}: 已预分配 {allocate_memory / (1024**3):.2f} GB / {total_memory / (1024**3):.2f} GB 显存")
        except RuntimeError as e:
            logger.warning(f"GPU {device_id}: 无法预分配全部显存，将使用动态分配。错误: {e}")
        
        # 设置CUDA分配器配置，减少碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        # 启用cudnn benchmark以优化性能
        torch.backends.cudnn.benchmark = True
        logger.info("GPU显存预分配完成，cudnn benchmark已启用")
    else:
        logger.info("未检测到可用GPU，跳过显存预分配")


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def additional_trdetr_losses(model_inputs, outputs, targets, opt):
    # TR-DETR only loss
    src_txt_mask,   src_vid_mask = model_inputs['src_txt_mask'], model_inputs['src_vid_mask']
    pos_mask =  targets['src_pos_mask'] 

    src_txt_ed, src_vid_ed =  outputs['src_txt_ed'], outputs['src_vid_ed']
    loss_align = CTC_Loss()
    loss_vid_txt_align = loss_align(src_vid_ed, src_txt_ed, pos_mask, src_vid_mask, src_txt_mask)

    src_vid_cls_ed = outputs['src_vid_cls_ed']
    src_txt_cls_ed = outputs['src_txt_cls_ed']
    loss_align_VTC = VTCLoss()
    loss_vid_txt_align_VTC = loss_align_VTC(src_txt_cls_ed, src_vid_cls_ed)

    loss = opt.VTC_loss_coef * loss_vid_txt_align_VTC + opt.CTC_loss_coef * loss_vid_txt_align
    return loss

def calculate_taskweave_losses(loss_dict, weight_dict, hd_log_var, mr_log_var):
    # TaskWeave only loss
    grouped_losses = {"loss_mr": [], "loss_hd": []}
    for k in loss_dict.keys():
        if k in weight_dict:
            if any(keyword in k for keyword in ["giou", "span", "label",'class_error']):
                grouped_losses["loss_mr"].append(loss_dict[k])
            elif "saliency" in k:
                grouped_losses["loss_hd"].append(loss_dict[k])
    loss_mr = sum(grouped_losses["loss_mr"])
    loss_hd = sum(grouped_losses["loss_hd"])
    # hd_log_var, mr_log_var = hd_log_var.to(loss_hd.device), mr_log_var.to(loss_mr.device)
    losses = 2 * loss_hd * torch.exp(-hd_log_var) + 1 * loss_mr * torch.exp(-mr_log_var) + hd_log_var + mr_log_var
    return losses


def run_multi_query_evaluation(opt, best_epoch):
    """训练结束后自动运行完整评估"""
    logger.info("="*50)
    logger.info("Training completed. Starting multi-query evaluation...")
    logger.info("="*50)
    
    # 构造评估命令
    model_path = os.path.join(opt.ckpt_filepath, "best.ckpt")
    results_base_dir = os.path.join(opt.results_dir, "multi_query_results")
    os.makedirs(results_base_dir, exist_ok=True)
    
    # 调用multi_query_evaluate.py脚本进行统一评估
    cmd = f"python scripts/multi_query_evaluate.py " \
          f"--model_path {model_path} " \
          f"--config_path {opt.yaml_path} " \
          f"--eval_path {opt.eval_path} " \
          f"--results_dir {results_base_dir} " \
          f"--eval_split_name {opt.eval_split_name}"
    
    logger.info(f"Running multi-query evaluation: {cmd}")
    result = os.system(cmd)
    
    if result != 0:
        logger.error("Multi-query evaluation failed!")
        return None
    
    # 查找最新生成的结果目录
    import glob as glob_module
    from datetime import datetime
    timestamp_dirs = glob_module.glob(os.path.join(results_base_dir, "evaluation_*"))
    if timestamp_dirs:
        # 按修改时间排序，获取最新的目录
        latest_dir = max(timestamp_dirs, key=os.path.getmtime)
        results_dir = latest_dir
    else:
        logger.warning("No evaluation results directory found")
        return None
    
    # 上传到wandb
    if opt.wandb_mode != "disabled":
        logger.info("Uploading all training and evaluation results to wandb...")
        
        # 1. 上传训练过程中的所有CSV文件
        training_csv_files = []
        for root, dirs, files in os.walk(opt.results_dir):
            for file in files:
                if file.endswith('.csv') and 'multi_query_results' not in root:
                    csv_path = os.path.join(root, file)
                    training_csv_files.append(csv_path)
                    wandb.save(csv_path)
                    logger.info(f"Uploaded training CSV: {csv_path}")
        
        # 2. 创建FINAL文件夹并上传所有最终评估结果
        final_dir = "FINAL_results"
        os.makedirs(final_dir, exist_ok=True)
        
        # 复制聚合结果到FINAL文件夹
        aggregate_csv = os.path.join(results_dir, "aggregated_results.csv")
        if os.path.exists(aggregate_csv):
            final_aggregate_csv = os.path.join(final_dir, "aggregated_results.csv")
            import shutil
            shutil.copy2(aggregate_csv, final_aggregate_csv)
            wandb.save(final_aggregate_csv)
            logger.info(f"Uploaded FINAL aggregated results: {final_aggregate_csv}")
            # 读取CSV并记录为表格，显著标识为FINAL
            df = pd.read_csv(final_aggregate_csv)
            wandb.log({"FINAL_multi_query_evaluation": wandb.Table(dataframe=df)})
        
        # 3. 复制所有多查询评估的详细metrics CSV到FINAL文件夹
        final_evaluation_files = []
        for csv_file in glob(os.path.join(results_dir, "metrics_*.csv")):
            final_csv_file = os.path.join(final_dir, os.path.basename(csv_file))
            shutil.copy2(csv_file, final_csv_file)
            wandb.save(final_csv_file)
            final_evaluation_files.append(final_csv_file)
            logger.info(f"Uploaded FINAL evaluation CSV: {final_csv_file}")
        
        # 4. 记录FINAL结果的关键指标
        if os.path.exists(aggregate_csv):
            df = pd.read_csv(aggregate_csv)
            # 为每个查询类型记录FINAL指标
            for _, row in df.iterrows():
                query_type = row['query_type']
                for col in df.columns:
                    if col != 'query_type' and col != 'epoch':
                        metric_name = f"FINAL_{query_type}_{col}"
                        wandb.log({metric_name: row[col]})
        
        # 5. 记录上传的文件统计信息
        total_csv_files = len(training_csv_files) + len(final_evaluation_files)
        wandb.log({
            "uploaded_files/training_csv_count": len(training_csv_files),
            "uploaded_files/FINAL_evaluation_csv_count": len(final_evaluation_files),
            "uploaded_files/total_csv_count": total_csv_files,
            "FINAL_evaluation_completed": True
        })
        
        logger.info(f"FINAL evaluation results uploaded. Total CSV files: {total_csv_files}")
        
        # 6. 清理本地FINAL文件夹
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
            logger.info(f"Cleaned up local FINAL folder: {final_dir}")
    
    logger.info(f"Multi-query evaluation completed. Results saved to: {results_dir}")
    return results_dir

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, teacher_model=None):
    batch_input_fn = cg_detr_prepare_batch_inputs  if opt.model_name == 'cg_detr' else prepare_batch_inputs
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    criterion.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    
    iter_modal_list = [None, "segment", "text"]
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):

        model_inputs, targets = batch_input_fn(batch[1], opt.device, opt, training=True)
        if "video_feat" in opt.post_ttt_input_type:
            relevant_video_feat = get_relevant_video_feat(batch_meta=batch[0], video_feat_dict=train_loader.dataset.video_feat_dict)
        else:
            relevant_video_feat = None
        if opt.has_teacher:
            iter_modal_idx = batch_idx%len(iter_modal_list)
            use_modal = iter_modal_list[iter_modal_idx]
        else:
            use_modal=None
        if teacher_model is not None:
            teacher_outputs = teacher_model(**model_inputs, batch_meta = batch[0], relevant_video_feat = relevant_video_feat, training=True, use_modal="segment_teacher")
        outputs = model(**model_inputs, batch_meta = batch[0], relevant_video_feat = relevant_video_feat, training=True, use_modal = use_modal)
        if teacher_model is not None:
            outputs['teacher_pred_logits'] = teacher_outputs['pred_logits']
            outputs['teacher_pred_spans'] = teacher_outputs['pred_spans']
            # outputs['teacher_saliency_scores'] = teacher_outputs['saliency_scores']
            outputs['teacher_query_feats'] = teacher_outputs['query_feats']
        
        loss_dict = criterion(outputs, targets, opt)        
        # remove reg_loss
        reg_loss = loss_dict['reg_loss'][1]
        # sal_loss = loss_dict['sal_loss'][1]
        loss_dict['reg_loss'] = loss_dict['reg_loss'][0]
        # loss_dict['sal_loss'] = loss_dict['sal_loss'][0]
        pos_cls_loss = loss_dict['cls_loss'][1]
        neg_cls_loss = loss_dict['cls_loss'][2]
        pos_cls_ratio = loss_dict['cls_loss'][3]
        neg_cls_ratio = loss_dict['cls_loss'][4]
        loss_dict['cls_loss'] = loss_dict['cls_loss'][0]

        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_dict["loss_overall"] = float(losses)
        
        loss_meters["loss_overall"].update(loss_dict["loss_overall"]) 
        loss_meters["cls_loss"].update(loss_dict['cls_loss'] * criterion.weight_dict['cls_loss'])
        if teacher_model is not None:
            loss_meters["dis_loss"].update(loss_dict['dis_loss'] * criterion.weight_dict['dis_loss'])
        loss_meters["cls_loss_pos"].update(pos_cls_loss, part=True) 
        loss_meters["cls_loss_neg"].update(neg_cls_loss, part=True) 
        loss_meters["pos_cls_ratio"].update(pos_cls_ratio) 
        loss_meters["neg_cls_ratio"].update(neg_cls_ratio) 
        loss_meters["reg_loss"].update(reg_loss* criterion.weight_dict['reg_loss'], part=True) 
        # loss_meters["sal_loss"].update(sal_loss* criterion.weight_dict['sal_loss'], part=True) 

        if batch_idx % opt.log_interval == 0:
            write_log(opt, epoch_i, loss_meters)
        
        # 记录训练指标到wandb (减少频率，只在log_interval时记录)
        if opt.wandb_mode != "disabled" and batch_idx % opt.log_interval == 0:
            wandb_log_dict = {
                f"train/{k}": v.avg for k, v in loss_meters.items()
            }
            wandb_log_dict["train/epoch"] = epoch_i + 1
            wandb_log_dict["train/learning_rate"] = optimizer.param_groups[0]['lr']
            wandb.log(wandb_log_dict)
        
def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt, teacher_model=None):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    collate_fn = cg_detr_start_end_collate if opt.model_name == 'cg_detr' else start_end_collate
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    # indices = np.random.choice(len(train_dataset), opt.bsz*4000, replace=False)
    # sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=opt.bsz,
        num_workers=0,
        shuffle=False,
        # pin_memory=True,
        # prefetch_factor=opt.bsz//4,
        # persistent_workers=True
        # sampler=sampler,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, teacher_model = teacher_model)
        lr_scheduler.step()

        if opt.model_ema:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval == 0 and (epoch_i + 1) > opt.start_eval_epoch:
        # if True:
            if opt.has_teacher:
                iter_modal_list = [None, "segment", "text"]
                iter_modal_list = ["text"]
            else:
                iter_modal_list = [None]
            for iter_modal in iter_modal_list:
                with torch.no_grad():
                    if opt.model_ema:
                        metrics, eval_loss_meters, latest_file_paths = \
                            eval_epoch(epoch_i, model_ema.module, val_dataset, opt, save_submission_filename, criterion,iter_modal=iter_modal)
                    else:
                        metrics, eval_loss_meters, latest_file_paths = \
                            eval_epoch(epoch_i, model, val_dataset, opt, save_submission_filename, criterion, iter_modal=iter_modal)

                write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode='val', iter_modal=iter_modal)     
                metricstocsv(epoch_i, metrics, opt.results_dir + f"metrics{iter_modal}.csv" if opt.results_dir[-1] == '/' else opt.results_dir +f"/metrics{iter_modal}.csv")       
                logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
                
                # 记录验证指标到wandb
                if opt.wandb_mode != "disabled":
                    wandb_log_dict = {
                        f"val/{k}": v for k, v in metrics["brief"].items()
                    }
                    wandb_log_dict.update({
                        f"val_loss/{k}": v.avg for k, v in eval_loss_meters.items()
                    })
                    if iter_modal:
                        wandb_log_dict[f"val_modal/{iter_modal}"] = 1
                    wandb_log_dict["val/epoch"] = epoch_i + 1
                    wandb.log(wandb_log_dict)
            
            if opt.dset_name == 'tvsum' or opt.dset_name == 'youtube_highlight':
                stop_score = metrics["brief"]["mAP"]
            else:
                stop_score = metrics["brief"]["On-Avg-R1@0.5_pos"]

        save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
        if stop_score > prev_best_score:
            prev_best_score = stop_score
            best_epoch = epoch_i
            logger.info("The checkpoint file has been updated.")
            rename_latest_to_best(latest_file_paths)
            
            # 上传best checkpoint到wandb
            if opt.wandb_mode != "disabled" and opt.get("wandb_log_model", True):
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="model",
                    description=f"Best model at epoch {epoch_i}"
                )
                artifact.add_file(os.path.join(opt.ckpt_filepath, f"epoch{epoch_i}.ckpt"))
                wandb.log_artifact(artifact)
                logger.info(f"Best model checkpoint uploaded to wandb at epoch {epoch_i}")
        
        if epoch_i == opt.n_epoch -1:
            best_file_path = os.path.join(opt.ckpt_filepath, f"epoch{best_epoch}.ckpt")
            new_file_path = os.path.join(opt.ckpt_filepath, "best.ckpt")
            os.rename(best_file_path, new_file_path)
    
    # 训练完成后运行多查询评估
    if not debug:
        multi_query_results_dir = run_multi_query_evaluation(opt, best_epoch)
    
    # 结束wandb run
    if opt.wandb_mode != "disabled":
        wandb.finish()

def main(opt, yaml_path, pretrained_model_path, domain, debug):
    # dataset & data loader
    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=opt.domain,
        data_path=opt.train_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=opt.a_feat_dirs if "a_feat_dirs" in opt else [],
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=opt.a_feat_types if "a_feat_types" in opt else None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
        segment_size=opt.segment_size,
        anchor_windows=opt.anchor_windows,
        use_online=opt.use_online,
        pos_threshold=opt.pos_threshold,
        label_file_path=opt.label_file_path,
        debug=debug,
        opt=opt,
        training=True,
    )

    train_dataset = CGDETR_StartEndDataset(**dataset_config) if opt.model_name == 'cg_detr' else StartEndDataset(**dataset_config)    
    copied_eval_config = copy.deepcopy(dataset_config)
    copied_eval_config.data_path = opt.eval_path
    copied_eval_config.q_feat_dir = opt.t_feat_dir_eval if "t_feat_dir_eval" in opt else opt.t_feat_dir
    copied_eval_config.training = False
    eval_dataset = CGDETR_StartEndDataset(**copied_eval_config) if opt.model_name == 'cg_detr' else StartEndDataset(**copied_eval_config)
    
    # prepare model
    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")

    # if opt.has_teacher:
    #     opt = BaseOptions().parse(yaml_path, domain)
    #     opt1 = opt
    #     opt1.use_lstm = False
    #     opt1.use_gru = False
    #     opt1.use_attn = False
    #     teacher_model, criterion, _, _ = setup_model(opt1)
    #     checkpoint = torch.load(opt1.teacher_model_path)
    #     teacher_model.load_state_dict(checkpoint["model"], strict=False)

    # prepare teacher model
    if opt.has_teacher:
        opt1 = BaseOptions().parse(yaml_path, domain, base_path="configs/base.yml")
        teacher_model, criterion, _, _ = setup_model(opt1)
        checkpoint = torch.load(opt.teacher_model_path)
        teacher_model.load_state_dict(checkpoint["model"], strict=False)
    # load checkpoint
    if pretrained_model_path is not None:
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint["model"])
        logger.info("Model checkpoint: {}".format(pretrained_model_path))
    count_parameters(model)
    logger.info("Start Training...")
    
    # start training
    if opt.has_teacher:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt, teacher_model = teacher_model)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='yaml config path for training. e.g., configs/qd_detr_qvhighlight.yml')
    parser.add_argument('--pretrained_model_path', type=str, help='saved model path', default=None)
    parser.add_argument('--domain', type=str, help='training domain for TVSum and YouTube Highlights . e.g., BK and dog. Note that they are not necessary for other datasets')
    parser.add_argument('--debug', type=bool, help='debug', default=False)
    parser.add_argument('--savecode', action='store_true', help='debug')
    args = parser.parse_args()
    yaml_path = args.config
    pretrained_model_path = args.pretrained_model_path
    domain = args.domain
    debug = args.debug

    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse(yaml_path, domain)
    opt["debug"] = debug
    opt.yaml_path = yaml_path  # 保存yaml路径供后续使用
    if debug:
        opt.results_dir = "exp/debug"
        opt.wandb_dir = "wandb/debug"
        # opt.eval_path = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/highlight_val_release_debug.jsonl"
        opt.ckpt_filepath = "exp/debug/checkpoint"
        opt.start_eval_epoch = 0
        opt.start_eval_epoch = 0
    if args.savecode:
        save_sh_n_codes(opt.results_dir)
        shutil.copy(args.config, opt.results_dir)
        shutil.copy("configs/base.yml", opt.results_dir)
    set_seed(opt.seed)
    
    # 预分配GPU显存
    allocate_full_gpu_memory()
    
    # 设置wandb模式
    if "wandb_mode" not in opt:
        opt.wandb_mode = "offline"
    os.environ["WANDB_MODE"] = opt.wandb_mode

    # 初始化wandb
    if opt.wandb_mode != "disabled":
        wandb_dir = opt.wandb_dir if not debug else "wandb/debug"
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        
        # 生成唯一的run名称，包含时间戳和配置信息
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用更有识别性的参数组合，包含关键超参数
        model_name = getattr(opt, 'model_name', 'unknown')
        dset_name = getattr(opt, 'dset_name', 'unknown')
        lr = getattr(opt, 'lr', 'unknown')
        bsz = getattr(opt, 'bsz', 'unknown')
        n_epoch = getattr(opt, 'n_epoch', 'unknown')
        # 构建包含关键超参数的run名称
        run_name = f"vanilla_{model_name}_{dset_name}_lr{lr}_bsz{bsz}_ep{n_epoch}_{timestamp}"
        
        # 添加更多标签来区分训练
        tags = ["training"]
        if domain:
            tags.append(f"domain_{domain}")
        if pretrained_model_path:
            tags.append("pretrained")
        if debug:
            tags.append("debug")
        
        wandb.init(
            dir=wandb_dir,
            project=opt.get("wandb_project", "online-vg"),
            entity=opt.get("wandb_entity", None),
            config=dict(opt),
            name=run_name,
            tags=tags,
            resume="allow"
        )
        logger.info(f"Wandb initialized in {opt.wandb_mode} mode with run name: {run_name}")
    
    main(opt, yaml_path, pretrained_model_path, domain, debug)
