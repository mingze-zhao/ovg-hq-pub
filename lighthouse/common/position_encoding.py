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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """
        Args:
            input_feat: (N, L, D)
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # (bsz, L, num_pos_feats*2)
        return pos_x  # .permute(0, 2, 1)  # (bsz, num_pos_feats*2, L)

class PositionEmbeddingSineOnlineSegment(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, posseg_len=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.posseg_len = posseg_len

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, shape (batch_size, L, d)
            mask: torch.tensor, shape (batch_size, L), 1代表有效位置

        Returns:
            pos_x: torch.tensor, shape (batch_size, L, num_pos_feats*2)
        """
        assert mask is not None, "mask should not be None"

        bsz, L = mask.shape
        device = x.device

        # 初始化 pos_x 为默认编码 [0, 1, 0, 1, ...]
        pos_x = torch.zeros(bsz, L, self.num_pos_feats, device=device)
        pos_x[:, :, 1::2] = 1  # 设置奇数维度为1

        for i in range(bsz):
            # 获取当前批次的mask
            current_mask = mask[i]  # shape: (L,)

            # 查找第一个有效的位置
            valid_indices = (current_mask == 1).nonzero(as_tuple=False)
            if len(valid_indices) == 0:
                continue  # 如果没有有效位置，跳过

            start = valid_indices[0].item()

            # 从第一个有效位置开始，以posseg_len为长度进行切分
            s = start
            while s < L:
                e = min(s + self.posseg_len, L)
                seg_mask = current_mask[s:e]

                if seg_mask.sum() == 0:
                    s = e
                    continue  # 如果segment中没有有效位置，跳过

                # 计算cumsum
                x_embed = seg_mask.cumsum(0).float()  # shape: (seg_len,)

                if self.normalize:
                    eps = 1e-6
                    x_embed = x_embed / (x_embed[-1] + eps) * self.scale

                # 计算dim_t
                dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

                # 计算pos_x
                pos = x_embed[:, None] / dim_t  # shape: (seg_len, num_pos_feats)
                pos = torch.stack((pos[:, 0::2].sin(), pos[:, 1::2].cos()), dim=2).flatten(1)  # shape: (seg_len, num_pos_feats*2)

                # 将计算得到的pos_x赋值到对应的位置
                pos_x[i, s:e, :] = pos

                s = e  # 继续下一个segment

        return pos_x
    
class PositionEmbeddingSineOnline(nn.Module):
    """
    实现在线的相对位置编码：
    当处理序列时，只能基于已经处理过的 Token（包括当前 Token）的信息来计算位置编码，
    不使用未来未见过的 Token 信息。
    输入:
        x: torch.tensor, (1, L, d)
        mask: torch.tensor, (1, L)，掩码，1 表示有效位置

    输出:
        pos_x: torch.tensor, (1, L, d)，与输入特征维度一致的位置编码
    """

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 位置编码的特征维度
        self.temperature = temperature      # 控制频率的缩放参数

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, (1, L, d)
            mask: torch.tensor, (1, L)

        Returns:
            pos_x: torch.tensor, (1, L, d)
        """
        bsz, seq_len, feature_dim = x.size()
        assert feature_dim >= self.num_pos_feats, "Feature dimension must be >= num_pos_feats."

        # 找到有效的 token 索引
        valid_positions = torch.nonzero(mask[0], as_tuple=True)[0]  # (有效位置数,)

        # 生成相对位置索引，以0开始递增，不再使用翻转从末尾计算
        # 这样每个位置的编码只依赖从头数到当前的位置，不涉及未来信息
        rel_positions = torch.arange(len(valid_positions), device=x.device).unsqueeze(-1)  # (有效位置数, 1)

        # 生成频率基数 dim_t
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 计算正弦和余弦编码
        rel_pos_enc = rel_positions / dim_t
        # 与传统正余弦编码保持一致的奇偶维度分配
        rel_pos_enc = torch.stack((rel_pos_enc[..., 0::2].sin(), rel_pos_enc[..., 1::2].cos()), dim=2).flatten(1)  

        # 将位置编码扩展到输入特征维度
        if feature_dim > self.num_pos_feats:
            padding = torch.zeros(len(valid_positions), feature_dim - self.num_pos_feats, device=x.device)
            rel_pos_enc = torch.cat([rel_pos_enc, padding], dim=-1)  # (有效位置数, feature_dim)

        # 初始化完整的输出，并填充有效位置编码
        pos_x = torch.zeros_like(x)
        pos_x[0, valid_positions, :] = rel_pos_enc

        return pos_x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    # elif args.position_embedding in ('v3', 'learned'):
    #     position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding == 'online':  # Deformable DETR (paper not released)
        position_embedding = PositionEmbeddingSineOnline(N_steps)
    elif args.position_embedding == 'onlinesegment':  # Deformable DETR (paper not released)
        position_embedding = PositionEmbeddingSineOnlineSegment(N_steps, normalize=True, posseg_len = args.posseg_len)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    txt_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=args.max_q_l,
            hidden_size=args.hidden_dim, dropout=args.input_dropout)
    return position_embedding, txt_pos_embed
