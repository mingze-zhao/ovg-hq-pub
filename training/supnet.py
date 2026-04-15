import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from training.config import BaseOptions
import time
import h5py
from tensorboardX import SummaryWriter
# from dataset import VideoDataSet, SuppressDataSet #-----------------------------------------------------
from lighthouse.common.online_suppress_net import SuppressNet
from lighthouse.common.loss_func import suppress_loss_func
from training.sup_dataset import SuppressDataSet
import argparse
from easydict import EasyDict
from training.cg_detr_dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs
from training.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from lighthouse.common.qd_detr import build_model as build_model_qd_detr
from lighthouse.common.online_qd_detr import build_model as build_model_online_qd_detr
from lighthouse.common.moment_detr import build_model as build_model_moment_detr
from lighthouse.common.cg_detr import build_model as build_model_cg_detr
from lighthouse.common.eatr import build_model as build_model_eatr
from lighthouse.common.tr_detr import build_model as build_model_tr_detr
from lighthouse.common.uvcom import build_model as build_model_uvcom
from lighthouse.common.taskweave import build_model as build_model_task_weave
from collections import OrderedDict, defaultdict
from lighthouse.common.utils.basic_utils import AverageMeter, save_sh_n_codes
from tqdm import tqdm, trange
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)



## ----------------------------------------------------------------------------------------------------------------
def train_one_epoch(opt, model, train_dataset, optimizer):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt['sup_batch_size'], shuffle=True,
                                                num_workers=4, pin_memory=True,drop_last=False)      
    epoch_cost = 0
    
    for n_iter,(input_data,label) in tqdm(enumerate(train_loader), total=(len(train_loader))):
        suppress_conf = model(input_data.cuda())
        
        loss = suppress_loss_func(label,suppress_conf)
        epoch_cost+= loss.detach().cpu().numpy()    
               
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
                
    return n_iter, epoch_cost
## ----------------------------------------------------------------------------------------------------------------
def eval_one_epoch(opt, model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=opt['sup_batch_size'], shuffle=False,
                                                num_workers=4, pin_memory=True,drop_last=False)   
    epoch_cost = 0
    
    for n_iter,(input_data,label) in tqdm(enumerate(test_loader), total=(len(test_loader))):
        suppress_conf = model(input_data.cuda())
        
        loss = suppress_loss_func(label,suppress_conf)
        epoch_cost+= loss.detach().cpu().numpy()    
               
    return n_iter, epoch_cost

## ----------------------------------------------------------------------------------------------------------------
def train(opt): 
    # writer = SummaryWriter()
    model = SuppressNet(opt).cuda()
    
    optimizer = optim.Adam( model.parameters(),lr=opt["sup_lr"],weight_decay = opt["sup_weight_decay"])     
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["sup_lr_step"])

    train_dataset = SuppressDataSet(opt,subset="train")      
    test_dataset = SuppressDataSet(opt,subset=opt['inference_subset'])
    
    for n_epoch in range(opt['n_epoch']):   
        n_iter, epoch_cost = train_one_epoch(opt, model, train_dataset, optimizer)
            
        # writer.add_scalars('sup_data/cost', {'train': epoch_cost/(n_iter+1)}, n_epoch)
        print("training loss(epoch %d): %f, lr - %f"%(n_epoch,
                                                                epoch_cost/(n_iter+1),
                                                                optimizer.param_groups[0]["lr"]) )
        
        scheduler.step()
        model.eval()
        
        n_iter, eval_cost = eval_one_epoch(opt, model,test_dataset)
        
        # writer.add_scalars('sup_data/eval', {'test': eval_cost/(n_iter+1)}, n_epoch)
        print("testing loss(epoch %d): %f"%(n_epoch,eval_cost/(n_iter+1)))
                    
        state = {'epoch': n_epoch + 1,
                    'state_dict': model.state_dict()}
        if not os.path.exists(opt["sup_checkpoint_path"]):
            os.makedirs(opt["sup_checkpoint_path"])
        torch.save(state, opt["sup_checkpoint_path"]+"/checkpoint_suppress.pth.tar" )
        if eval_cost < model.best_loss:
            model.best_loss = eval_cost
            torch.save(state, opt["sup_checkpoint_path"]+"/ckp_best_suppress.pth.tar" )
            
        model.train()
                
    # writer.close()
    return 

def eval_frame(opt, model, dataset, criterion=None):
    collate_fn = cg_detr_start_end_collate if opt.model_name == 'cg_detr' else start_end_collate
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        prefetch_factor=opt.bsz//8,
        shuffle=False,
        pin_memory=True,
    )
    
    if "use_online" in opt:
        labels_cls={}
        labels_reg={}
        output_cls={}
        output_reg={}  
        for i in range(len(eval_loader.dataset.data)):
            qid = str(eval_loader.dataset.data[i]['qid'])
            labels_cls[qid]=[]
            labels_reg[qid]=[]
            output_cls[qid]=[]
            output_reg[qid]=[]

        batch_input_fn = prepare_batch_inputs
        loss_meters = defaultdict(AverageMeter)

        for n_iter, batch in tqdm(enumerate(eval_loader), desc="compute st ed scores", total=len(eval_loader)):
            query_meta = batch[0]
            model_inputs, targets = batch_input_fn(batch[1], opt.device)

            outputs = model(**model_inputs)
            # if criterion:
            #     loss_dict = criterion(outputs, targets, opt)
            #     # remove reg_loss
            #     reg_loss = loss_dict['reg_loss'][1]
            #     loss_dict['reg_loss'] = loss_dict['reg_loss'][0]
            #     pos_cls_loss = loss_dict['cls_loss'][1]
            #     neg_cls_loss = loss_dict['cls_loss'][2]
            #     pos_cls_ratio = loss_dict['cls_loss'][3]
            #     neg_cls_ratio = loss_dict['cls_loss'][4]
            #     loss_dict['cls_loss'] = loss_dict['cls_loss'][0]

            #     weight_dict = criterion.weight_dict
            #     losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            #     loss_dict["loss_overall"] = float(losses)
            #     for k, v in loss_dict.items():
            #         if k != 'reg_loss':
            #             loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))
            #         else:
            #             loss_meters[k].update(reg_loss, part=True) 
            #     loss_meters["loss_overall"].update(loss_dict["loss_overall"]) 
            #     loss_meters["cls_loss"].update(loss_dict['cls_loss']/opt.eos_coef)   
            #     loss_meters["cls_loss_pos"].update(pos_cls_loss, part=True) 
            #     loss_meters["cls_loss_neg"].update(neg_cls_loss/opt.eos_coef, part=True) 
            #     loss_meters["pos_cls_ratio"].update(pos_cls_ratio) 
            #     loss_meters["neg_cls_ratio"].update(neg_cls_ratio) 
            #     loss_meters["reg_loss"].update(reg_loss, part=True) 

            # compose predictions
            pred_spans = outputs["pred_spans"].cpu()  # (bsz, #queries, 2)
            prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)

            for b in range(0, batch[1]['video_feat'][0].shape[0]):
                qid, vid, st, ed, data_idx, line_idx = eval_loader.dataset.inputs[n_iter*opt['bsz']+b]
                output_cls[str(qid)]+=[prob[b,:].detach().cpu().numpy()]
                output_reg[str(qid)]+=[pred_spans[b,:].detach().cpu().numpy()]
                labels_cls[str(qid)]+=[targets["cls_labels"][b,:].cpu().numpy()]
                labels_reg[str(qid)]+=[targets["reg_labels"][b,:].cpu().numpy()]

        for i in range(len(eval_loader.dataset.data)):
            qid = str(eval_loader.dataset.data[i]['qid'])
            # labels_cls[qid]=np.stack(labels_cls[qid], axis=0)
            # labels_reg[qid]=np.stack(labels_reg[qid], axis=0)
            output_cls[qid]=np.stack(output_cls[qid], axis=0)
            output_reg[qid]=np.stack(output_reg[qid], axis=0)

        # mr_res = eval_map_nms(opt, eval_loader.dataset, output_cls, output_reg, labels_cls, labels_reg)

    return output_cls, output_reg, labels_cls, labels_reg

## ----------------------------------------------------------------------------------------------------------------
def test(opt): 
    model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    dataset = SuppressDataSet(opt,subset=opt['inference_subset'])
    
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt['batch_size'], shuffle=False,
                                                num_workers=4, pin_memory=True,drop_last=False)   
    labels={}
    output={}                                   
    for video_name in dataset.video_list:
        labels[video_name]=[]
        output[video_name]=[]
        
    for n_iter,(input_data,label) in tqdm(enumerate(test_loader)):
        suppress_conf = model(input_data.cuda())
          
        for b in range(0,input_data.size(0)):
            video_name, idx = dataset.inputs[n_iter*opt['batch_size']+b]
            output[video_name]+=[suppress_conf[b,:].detach().cpu().numpy()]
            labels[video_name]+=[label[b,:].numpy()]
        
    for video_name in dataset.video_list:
        labels[video_name]=np.stack(labels[video_name], axis=0)
        output[video_name]=np.stack(output[video_name], axis=0)

    outfile = h5py.File(opt['suppress_result_file'], 'w')
    
    for video_name in dataset.video_list:
        o=output[video_name]
        l=labels[video_name]
        
        dset_pred = outfile.create_dataset(video_name+'/pred', o.shape, maxshape=o.shape, chunks=True, dtype=np.float32)
        dset_pred[:,:] = o[:,:]  
        dset_label = outfile.create_dataset(video_name+'/label', l.shape, maxshape=l.shape, chunks=True, dtype=np.float32)
        dset_label[:,:] = l[:,:]  
    outfile.close()
    print('complete')

def build_model(opt):
    if opt.model_name == 'qd_detr':
        model, criterion = build_model_qd_detr(opt)
    if opt.model_name == 'online_qd_detr':
        model, criterion = build_model_online_qd_detr(opt)
    elif opt.model_name == 'moment_detr':
        model, criterion = build_model_moment_detr(opt)
    elif opt.model_name == 'cg_detr':
        model, criterion = build_model_cg_detr(opt)
    elif opt.model_name == 'eatr':
        model, criterion = build_model_eatr(opt)
    elif opt.model_name == 'tr_detr':
        model, criterion = build_model_tr_detr(opt)
    elif opt.model_name == 'uvcom':
        model, criterion = build_model_uvcom(opt)
    elif opt.model_name == 'taskweave':
        model, criterion = build_model_task_weave(opt)
    else:
        raise NotImplementedError
    
    return model, criterion

def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    # logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)

    if opt.device == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    return model, criterion, optimizer, lr_scheduler

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

def make_dataset(opt): 
    # model = MYNET(opt).cuda()
    model, criterion, _, _ = setup_model(opt)
    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint["model"])
    # checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    # base_dict=checkpoint['state_dict']
    # model.load_state_dict(base_dict)
    model.eval()
    criterion.eval()

    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=opt.domain,
        data_path=opt.eval_path, ##
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
        subset="val", ##
        debug=opt.debug,
    )
    
    dataset = CGDETR_StartEndDataset(**dataset_config) if opt.model_name == 'cg_detr' else StartEndDataset(**dataset_config)    

    # dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    
    output_cls, output_reg, labels_cls, labels_reg = eval_frame(opt, model, dataset, criterion)
    
    # proposal_dict=[]
    
    outfile = h5py.File(opt['suppress_label_file'].format(opt['inference_subset']), 'w')
    
    result_dict={}
    proposal_dict=[]
    
    anchors=opt['anchor_windows']
    unit_size = opt['segment_size']
                                            
    for i in range(len(dataset.data)):
        qid = str(dataset.data[i]['qid'])
        vid = dataset.data[i]['vid']
        if dataset.dset_name == "qvhighlight":
            frame_to_time = 100.0* 2
        
        duration = int(dataset.data[i]['duration'] / 2)
        for idx in range(0,duration):
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
                if ed>int(dataset.data[i]['duration'] / 2):
                    ed=int(dataset.data[i]['duration'] / 2)
                tmp_dict={}
                tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                tmp_dict["score"]= cls_anc[anc_idx][0]*1.0
                tmp_dict["gentime"]= idx*frame_to_time/100.0
                proposal_anc_dict.append(tmp_dict)
            
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=0.3) 
            proposal_dict+=proposal_anc_dict
        
        nms_dict=non_max_suppression(proposal_dict, overlapThresh=opt['nms_threshold'])
               
        input_table = np.zeros((duration,unit_size,1), dtype=np.float32)
        label_table = np.zeros((duration,1), dtype=np.float32)
        
        for proposal in proposal_dict:
            idx = int(proposal["gentime"] / 2) 
            conf = proposal["score"]
            # cls = proposal["label"]
            for i in range(0,unit_size):
                if idx+i < duration:
                    input_table[idx+i,unit_size-1-i,0]=conf
        
        for proposal in nms_dict:
            idx = int(proposal["gentime"] / 2)
            # cls = proposal["label"]
            label_table[idx:idx+3,0]=1
        
        dset_input_table = outfile.create_dataset(qid+'/input', input_table.shape, maxshape=input_table.shape, chunks=True, dtype=np.float32)
        dset_label_table = outfile.create_dataset(qid+'/label', label_table.shape, maxshape=label_table.shape, chunks=True, dtype=np.float32)
        
        dset_input_table[:]=input_table
        dset_label_table[:]=label_table
        
        proposal_dict=[]
    
    print('complete')
    return
    
##  
def main(opt):
    if opt['mode'] == 'train':
        train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'make':
        make_dataset(opt)
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='yaml config path for training. e.g., configs/qd_detr_qvhighlight.yml')
    parser.add_argument('--domain', type=str, help='training domain for TVSum and YouTube Highlights . e.g., BK and dog. Note that they are not necessary for other datasets')
    parser.add_argument('--debug', type=bool, help='debug', default=False)
    args = parser.parse_args()
    yaml_path = args.config
    domain = args.domain
    opt = BaseOptions().parse(yaml_path, domain)
    opt.debug = args.debug
    if args.debug:
        opt.results_dir = "/mnt/data/jiaqi/online-vg/exp/debug"
        opt.wandb_dir = "/mnt/data/jiaqi/online-vg/wandb/debug"
    save_sh_n_codes(opt.sup_checkpoint_path)
    if opt['seed'] >= 0:
        seed = opt['seed'] 
        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
          
    main(opt)
    while(opt['wterm']):
        pass
