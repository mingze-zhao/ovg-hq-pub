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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from lighthouse.common.online_qd_detr_transformer import build_transformer
from lighthouse.common.position_encoding import build_position_encoding
from lighthouse.common.TTT import TTT, TTTConfig, TTT_Transformer, LSTM, Attn

import numpy as np
    
class Online_QDDETR(nn.Module):
    """ QD DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=True, max_v_l=75, span_loss_type="l1", 
                 use_txt_pos=False, n_input_proj=2, aud_dim=0, opt=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.opt = opt
        self.segment_size = opt.segment_size
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries,2)

        if opt.dec_predefined_anchor:
            output = torch.tensor(self.calculate_anchor_centers_and_widths(opt.segment_size, opt.anchor_windows))
            self.query_embed.weight.data.copy_(output)

        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim

        # Saliency
        # self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        # self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        # self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        # self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        if "image" in opt.query_type:
            self.input_img_proj = nn.Sequential(*[
                LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])

        if "segment" in opt.query_type:
            self.input_sgm_proj = nn.Sequential(*[
                LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])

        # Multi-modal query
        if opt.query_type == "image_text":
            self.img_token = torch.nn.Parameter(torch.randn(txt_dim))
            self.txt_token = torch.nn.Parameter(torch.randn(txt_dim))
        if opt.query_type == "segment_text":
            self.sgm_token = torch.nn.Parameter(torch.randn(vid_dim + aud_dim))
            self.txt_token = torch.nn.Parameter(torch.randn(txt_dim))

        ## ttt process output
        if "mr" in opt.post_ttt_task:
            if opt.post_ttt_layer_type == "ttt":
                if opt.use_lstm:
                    configuration = TTTConfig(hidden_size=256, intermediate_size=256, num_hidden_layers=1, num_attention_heads=1, mini_batch_size=self.num_queries)
                    self.post_ttt_mr = nn.ModuleList([LSTM(configuration, 256) for i in range(opt.post_ttt_layer_num)])
                    print("----LSTM----")
                elif opt.use_att:
                    configuration = TTTConfig(hidden_size=256, intermediate_size=256, num_hidden_layers=1, num_attention_heads=1, mini_batch_size=self.num_queries)
                    self.post_ttt_mr = nn.ModuleList([Attn(configuration, 256) for i in range(opt.post_ttt_layer_num)])
                    print("----ATTT----")
                else:
                    configuration = TTTConfig(hidden_size=256, intermediate_size=256, num_hidden_layers=1, num_attention_heads=1, mini_batch_size=self.num_queries)
                    self.post_ttt_mr = nn.ModuleList([TTT(configuration, i) for i in range(opt.post_ttt_layer_num)])
            elif opt.post_ttt_layer_type == "transformer_ttt":
                configuration = TTTConfig(hidden_size=hidden_dim, num_hidden_layers=1, num_attention_heads=1, mini_batch_size=self.num_queries)
                self.post_ttt_mr = nn.ModuleList([TTT_Transformer(configuration, i) for i in range(opt.post_ttt_layer_num)])
            if opt.post_ttt_input_fusion_type in ["4->D", "4->D||D", "4->D+D"]:
                if opt.post_ttt_input_fusion_type =="4->D||D":
                    self.ttt_input_fuse_proj = nn.Linear(len(opt.post_ttt_input_type)*hidden_dim, hidden_dim)
                    # self.ttt_input_fuse_norm = nn.LayerNorm(normalized_shape=hidden_dim)
                self.ttt_input_proj = nn.Linear(4, hidden_dim)
                self.ttt_output_proj = nn.Linear(hidden_dim, 4)

        if "hd" in opt.post_ttt_task:
            if opt.post_ttt_layer_type == "ttt":
                configuration = TTTConfig(hidden_size=1, num_hidden_layers=1, num_attention_heads=1, mini_batch_size=self.segment_size)
                self.post_ttt_hd = nn.ModuleList([TTT(configuration, i) for i in range(opt.post_ttt_layer_num)])
            elif opt.post_ttt_layer_type == "transformer_ttt":
                configuration = TTTConfig(hidden_size=1, num_hidden_layers=1, num_attention_heads=1, mini_batch_size=self.segment_size)
                self.post_ttt_hd = nn.ModuleList([TTT_Transformer(configuration, i) for i in range(opt.post_ttt_layer_num)])
        
        self.anchor_windows = torch.tensor(opt.anchor_windows).cuda()



    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_sgm=None, src_sgm_mask=None, src_sgm_teacher=None, src_sgm_mask_teacher=None, src_txt_teacher=None, src_txt_mask_teacher=None, src_img=None, src_img_mask=None, src_aud=None, src_aud_mask=None, batch_meta=None, relevant_video_feat=None, input=None, training=False, window_st_ed=None, use_modal=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in 
                               [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        if use_modal == "segment_teacher":
            src_sgm = src_sgm_teacher
            src_sgm_mask = src_sgm_mask_teacher
            src_txt = src_txt_teacher
            src_txt_mask = src_txt_mask_teacher
            use_modal = None

        if self.opt.multi_query and (use_modal is None):
            if src_img is not None:
                img_token = self.img_token.reshape([1, 1, self.img_token.size(0)]).repeat(src_vid.shape[0], 1, 1)
                txt_token = self.txt_token.reshape([1, 1, self.txt_token.size(0)]).repeat(src_vid.shape[0], 1, 1)
                src_img = torch.cat([img_token, src_img], dim=1)
                src_txt = torch.cat([txt_token, src_txt], dim=1)
                src_img = self.input_img_proj(src_img)
                src_txt = self.input_txt_proj(src_txt)
                src_txt = torch.cat([src_img, src_txt], dim=1)
                mask_ = torch.tensor([[True]]).to(src_vid.device).repeat(src_vid.shape[0], 3)
                src_txt_mask = torch.cat([mask_, src_txt_mask], dim=1).bool()

            if src_sgm is not None:
                sgm_token = self.sgm_token.reshape([1, 1, self.sgm_token.size(0)]).repeat(src_vid.shape[0], 1, 1)
                txt_token = self.txt_token.reshape([1, 1, self.txt_token.size(0)]).repeat(src_vid.shape[0], 1, 1)
                src_sgm = torch.cat([sgm_token, src_sgm], dim=1)
                src_txt = torch.cat([txt_token, src_txt], dim=1)
                src_sgm = self.input_sgm_proj(src_sgm)
                src_txt = self.input_txt_proj(src_txt)
                src_txt = torch.cat([src_sgm, src_txt], dim=1)
                mask_ = torch.tensor([[True]]).to(src_vid.device).repeat(src_vid.shape[0], src_sgm.size(1)+1)
                src_txt_mask = torch.cat([mask_, src_txt_mask], dim=1).bool()
        elif self.opt.multi_query and (use_modal is not None):
            if use_modal == "segment":
                src_txt = self.input_sgm_proj(src_sgm)
                src_txt_mask = torch.tensor([[True]]).to(src_vid.device).repeat(src_vid.shape[0], src_sgm.size(1))
            elif use_modal == "text":
                src_txt = self.input_txt_proj(src_txt)
        else:
            if "image" in self.opt.query_type:
                src_txt = self.input_img_proj(src_txt)
            elif "segment" in self.opt.query_type:
                src_txt = self.input_sgm_proj(src_txt)
            else:
                src_txt = self.input_txt_proj(src_txt)
                
        src_vid = self.input_vid_proj(src_vid) #bsz, L_vid, d
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)

        # get mask
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        # get position embedding
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d) ## 不能用, 因为online的setting下，不知道序列有多长
        # pos_vid = torch.zeros_like(src_vid).cuda()
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1) # (bsz, L_vid+L_txt, d)

        # for saliency token
        # mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        # mask = torch.cat([mask_, mask], dim=1)
        # src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        # src = torch.cat([src_, src], dim=1)
        # pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        # pos = torch.cat([pos_, pos], dim=1)

        #forward cross-attention transformer encoder, self-attention transformer encoder, transformer decoder
        hs, reference, memory = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=src_vid.shape[1]) # video length:16, pos全部一样, self.query_embed.weight: (num_queries, 2)不同， src: (bsz, L_vid+L_txt+1, d), mask: (bsz, L_vid+L_txt)
        # hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=src_vid.shape[1]) # video length:16, pos全部一样, self.query_embed.weight: (num_queries, 2)不同， src: (bsz, L_vid+L_txt+1, d), mask: (bsz, L_vid+L_txt)

        # hd_output
        # saliency_scores = (torch.sum(self.saliency_proj1(memory) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        # moment_ouput
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes) 
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        # if self.span_loss_type == "l1":
        outputs_coord = outputs_coord.sigmoid()

        # get prediction, query feature, video feature
        outputs_class_last = outputs_class[-1] # [4096, 5, 2]
        outputs_coord_last =  outputs_coord[-1] #[4096, 5, 2]

        if "mr" in self.opt.post_ttt_task:
            # if "query_feat" in self.opt.post_ttt_input_type:
            query_feature_last = hs[-1] # [4096, 5, 256]
            if "video_feat" in self.opt.post_ttt_input_type:
                anchors = self.anchor_windows
                video_features = torch.zeros_like(query_feature_last)
                for i in range(len(input)):
                    input_st = window_st_ed[i][0]
                    input_ed = window_st_ed[i][1]
                    reg_anc = outputs_coord_last[i]
                    ed = input_ed + anchors * reg_anc[:,0]
                    length = anchors * torch.exp(reg_anc[:,1])
                    st= ed-length
                    st = torch.round(torch.max(input_st, st)).int() - input_st
                    ed = torch.round(torch.min(input_ed, ed)).int() - input_st

                    ed = torch.where(ed <= st, st + 1, ed)

                    # 限制 st 和 ed 在 [0, 16] 范围内
                    ed = torch.clamp(ed, 0, 16)

                    # 如果 ed 仍然小于或等于 st，调整 st
                    st = torch.where(ed <= st, ed - 1, st)

                    range_tensor = torch.arange(16, device='cuda:0').unsqueeze(0)
                    mask = (range_tensor >= st.unsqueeze(1)) & (range_tensor < ed.unsqueeze(1))
                    memory_ = memory[i].repeat(5, 1, 1)
                    mask_expanded = mask.unsqueeze(-1).expand_as(memory_)
                    masked_memory_ = memory_.masked_fill(~mask_expanded, float('-inf'))
                    video_features[i,:] = torch.max(masked_memory_, dim=1)[0]
            
            # compose ttt input
            outputs_for_ttt = torch.cat((outputs_class_last, outputs_coord_last), dim=2)
            outputs_for_ttt = self.ttt_input_proj(outputs_for_ttt)
            sample_num = int(outputs_for_ttt.size(0)/self.opt.sequence_length) if training else self.opt.eval_bsz

            outputs_for_ttt = outputs_for_ttt.reshape(sample_num, -1, outputs_for_ttt.size(-1)) # [bsz*sgm_num, num_queries,D]
            if "query_feat" in self.opt.post_ttt_input_type:
                query_feature_last = query_feature_last.reshape(sample_num, -1, query_feature_last.size(-1))
            if "video_feat" in self.opt.post_ttt_input_type:
                video_features = video_features.reshape(sample_num, -1, video_features.size(-1))

            if self.opt.post_ttt_input_fusion_type == "4->D||D":
                if self.opt.post_ttt_input_type == ["pred", "query_feat"]:
                    outputs_for_ttt = torch.cat((outputs_for_ttt, query_feature_last), dim=2)
                    outputs_for_ttt = self.ttt_input_fuse_proj(outputs_for_ttt)
                elif self.opt.post_ttt_input_type == ["pred", "video_feat"]:
                    outputs_for_ttt = torch.cat((outputs_for_ttt, video_features), dim=2)
                    outputs_for_ttt = self.ttt_input_fuse_proj(outputs_for_ttt)
                elif self.opt.post_ttt_input_type == ["pred", "query_feat", "video_feat"]:
                    outputs_for_ttt = torch.cat((outputs_for_ttt, query_feature_last, video_features), dim=2)
                    outputs_for_ttt = self.ttt_input_fuse_proj(outputs_for_ttt)

            elif self.opt.post_ttt_input_fusion_type == "4->D+D":
                if self.opt.post_ttt_input_type == ["pred", "query_feat"]:
                    outputs_for_ttt = outputs_for_ttt + query_feature_last
                elif self.opt.post_ttt_input_type == ["pred", "video_feat"]:
                    outputs_for_ttt = outputs_for_ttt + video_features
                elif self.opt.post_ttt_input_type == ["pred", "query_feat", "video_feat"]:
                    outputs_for_ttt = outputs_for_ttt + video_features + query_feature_last

            # forward ttt
            for layer in self.post_ttt_mr:
                outputs_for_ttt, _= layer(outputs_for_ttt)

            if self.opt.post_ttt_input_fusion_type in ["4->D", "4->D||D", "4->D+D"]:
                outputs_for_ttt = self.ttt_output_proj(outputs_for_ttt)

            # decomposed ttt output
            outputs_for_ttt = outputs_for_ttt.reshape(-1, self.opt.num_queries, 4)
            outputs_class_last = outputs_for_ttt[..., :2]
            outputs_coord_last = outputs_for_ttt[..., 2:]
        
        # if "hd" in self.opt.post_ttt_task:
        #     saliency_scores = saliency_scores.reshape(sample_num, -1, 1)
        #     for layer in self.post_ttt_hd:
        #         saliency_scores, _= layer(saliency_scores, last_mini_batch_params_dict=None)
        #     saliency_scores = saliency_scores.squeeze(-1).reshape(-1, self.opt.segment_size)
 
        out = {'pred_logits': outputs_class_last, 'pred_spans': outputs_coord_last}
        # out["saliency_scores"] = saliency_scores
        out["query_feats"] =  hs
        
        # if torch.isnan(outputs_class_last).any() or torch.isnan(outputs_coord_last).any() or torch.isnan(out["saliency_scores"]).any():
            # print("nan in outputs_class_last")
            # exit(-1)
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        out["video_mask"] = src_vid_mask
        
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out

    def calculate_anchor_centers_and_widths(self, segment_length, anchors):
        results = []
        for anchor in anchors:
            # 计算起始位置
            start = segment_length - anchor
            # 计算中心
            center = (start + (start + anchor)) / 2
            # 归一化中心到0-1范围
            normalized_center = center / segment_length
            # 归一化宽度到0-1范围
            normalized_width = anchor / segment_length
            
            results.append([normalized_center, normalized_width])
        
        return results

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight_dict, eos_coef, losses, 
                 span_loss_type, max_v_l, saliency_margin=1, args=None):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        self.focal = focal_loss(alpha=args.alpha, gamma=args.gamma)

    def loss_cls(self, outputs, targets, opt):
        cls_labels = targets['cls_labels']
        pos_idx = (targets['cls_labels'][:,:,0]==1)
        neg_idx = (targets['cls_labels'][:,:,1]==1)
        if opt.use_focal_loss:
            loss_ce = self.focal(outputs['pred_logits'], (cls_labels[:,:,1]==1).long() )
            loss_ce_pos = self.focal(outputs['pred_logits'][pos_idx], torch.zeros(pos_idx.sum(),1).long().cuda().squeeze(1))
            loss_ce_neg = self.focal(outputs['pred_logits'][neg_idx], torch.ones(neg_idx.sum(),1).long().cuda().squeeze(1))
            cls_pos_ratio = loss_ce_pos.sum() / (loss_ce_pos.sum() +loss_ce_neg.sum())
            cls_neg_ratio = loss_ce_neg.sum() / (loss_ce_pos.sum() +loss_ce_neg.sum())
        else:
            loss_ce = F.cross_entropy(outputs['pred_logits'].reshape(-1,2), (cls_labels[:,:,1]==1).long().reshape(-1), self.empty_weight, reduction="none")
            loss_ce_pos = self.empty_weight[0] * F.cross_entropy(outputs['pred_logits'][pos_idx].reshape(-1,2), torch.zeros(pos_idx.sum(),1).long().cuda().squeeze(1), reduction="none")
            loss_ce_neg = self.empty_weight[1] * F.cross_entropy(outputs['pred_logits'][neg_idx].reshape(-1,2), torch.ones(neg_idx.sum(),1).long().cuda().squeeze(1), reduction="none")
            cls_pos_ratio = loss_ce_pos.sum() / (loss_ce_pos.sum() +loss_ce_neg.sum())
            cls_neg_ratio = loss_ce_neg.sum() / (loss_ce_pos.sum() +loss_ce_neg.sum())
        losses = {}
        losses['cls_loss'] = (loss_ce.mean(), loss_ce_pos, loss_ce_neg, cls_pos_ratio, cls_neg_ratio)
        return losses

    def loss_dis(self, outputs, targets, opt):
        # logits
        kd_loss = 0
        
        if "logits" in opt.distillation_pos:
            student_logits = outputs['pred_logits']
            teacher_logits = outputs['teacher_pred_logits']
            pred_probs_student = F.softmax(student_logits / opt.temperature, dim=-1)
            pred_probs_teacher = F.softmax(teacher_logits / opt.temperature, dim=-1)
            kl_loss = F.kl_div(F.log_softmax(pred_probs_student / opt.temperature, dim=-1), 
                    pred_probs_teacher, reduction='batchmean')
            kd_loss += kl_loss

        if "spans" in opt.distillation_pos:
            student_spans = outputs['pred_spans']
            teacher_spans = outputs['teacher_pred_spans']
            mse_loss_spans = F.mse_loss(student_spans, teacher_spans)
            kd_loss += mse_loss_spans

        if "query_feat_0" in opt.distillation_pos:
            student_query_feat_0 = outputs['query_feats'][0]
            teacher_query_feat_0 = outputs['teacher_query_feats'][0]
            mse_loss_layer0 = F.mse_loss(student_query_feat_0, teacher_query_feat_0)
            kd_loss += mse_loss_layer0

        if "query_feat_1" in opt.distillation_pos:
            student_query_feat_1 = outputs['query_feats'][1]
            teacher_query_feat_1 = outputs['teacher_query_feats'][1]
            mse_loss_layer1 = F.mse_loss(student_query_feat_1, teacher_query_feat_1)
            kd_loss += mse_loss_layer1
        
        kd_loss /= len(opt.distillation_pos)

        losses = {}
        losses['dis_loss'] = (kd_loss)
        # query_feat
        return losses
    
    def loss_reg(self, outputs, targets, opt):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        # if outputs['pred_spans'].shape[0] != 1024:
        #     print(1)
        reg_labels = targets['reg_labels']
        idx = (targets['reg_labels'][:,:,1]>-1000)
        reg_loss = F.l1_loss(
            outputs['pred_spans'][idx], reg_labels[idx], reduction='none')
        losses = {}
        if reg_loss.nelement() == 0:
            losses['reg_loss'] = (torch.tensor(0).cuda(), reg_loss)
        else:
            losses['reg_loss'] = (reg_loss.mean(), reg_loss.mean(dim=1))
        return losses

    def loss_sal(self, outputs, targets, opt):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        losses = {}
        sal_labels = targets['sal_labels']
        saliency_scores = outputs['saliency_scores']
        idx = (sal_labels[:,0]!=-1)
        saliency_scores = saliency_scores[idx]
        # if saliency_scores.nelement() == 0:
        #     losses['sal_loss'] = (torch.tensor(0).cuda(), 0)
        #     return losses
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        sal_labels = sal_labels[idx]
        neg_scores = saliency_scores[batch_indices, sal_labels[:,0].int()]
        pos_scores = saliency_scores[batch_indices, sal_labels[:,1].int()]
        sal_loss = torch.clamp(opt.saliency_margin + neg_scores - pos_scores, min=0)
        losses['sal_loss'] = (sal_loss.mean(), sal_loss)
        return losses

    def get_loss(self, loss, outputs, targets, opt, **kwargs):
        if opt.has_teacher:
            loss_map = {
                "cls": self.loss_cls,
                "reg": self.loss_reg,
                "sal": self.loss_sal,
                "dis": self.loss_dis,
            }
        else:
            loss_map = {
                "cls": self.loss_cls,
                "reg": self.loss_reg,
                "sal": self.loss_sal,
            }          
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, opt, **kwargs)

    def forward(self, outputs, targets, opt):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses_target = self.losses
        losses = {}
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, opt))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         losses_target = self.losses

        #         for loss in losses_target:
        #             kwargs = {}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        labels = labels.to(preds.device)
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        return loss

def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = Online_QDDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim if "a_feat_dim" in args else 0,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        opt=args,
    )

    weight_dict = {"reg_loss": args.reg_loss_coef,
                   "cls_loss": args.cls_loss_coef,
                   "sal_loss": args.sal_loss_coef,
                   "dis_loss": args.dis_loss_coef,
                   }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = args.online_loss
    criterion = SetCriterion(
        weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, span_loss_type=args.span_loss_type, 
        max_v_l=args.max_v_l,
        args=args
    )
    criterion.to(device)
    return model, criterion

