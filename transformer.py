"""
Transformer for EEG classification
"""


import argparse
import os
import numpy as np
import math
import glob
import random


import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.models import vgg19

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from torchsummary import summary

import matplotlib.pyplot as plt

seed_n = np.random.randint(500)#192 37
# seed_n =236# 87#96#350#212 #43#206#459  #264 #236 is best when expansion is 50
print('seed is ' + str(seed_n))

random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)


class PatchEmbedding(nn.Module):
    def __init__(self,  patch_size: int = 10, emb_size: int = 10, img_size: int = 1000):
        self.patch_size = patch_size
        super().__init__()
        self.channel_num=42
        self.in_channels=1
        self.spatial = nn.Sequential(
            nn.Conv2d(self.in_channels, 1, (self.channel_num, 1), (1, 1)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(2, 2, (22, 1), (1, 1)),
            # nn.BatchNorm2d(2),
            # nn.LeakyReLU(0.2),
            # nn.ELU(),
        )
        self.projection = nn.Sequential(
            # nn.Conv2d(1, emb_size, (1, 5), stride=(1, 5)),
            nn.Conv2d(self.in_channels, emb_size, (self.channel_num, 12), stride=(1, 4)),# 4
            # nn.MaxPool2d( kernel_size=(1,5), stride=(1,5)),
            Rearrange('b e (h) (w) -> b (h w) e'),

        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        # x = self.spatial(x)
        # x=rearrange(x,'b o c s -> b o (s c)')
        # x = rearrange(x, 'b o (h s) -> b o h s',h=1)
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # cls
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        # x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 10, num_heads: int = 5, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 10,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 10, n_classes: int = 3):
        # super().__init__(
        #     Reduce('b n e -> b e', reduction='mean'),
        #     nn.LayerNorm(emb_size),
        #     nn.Linear(emb_size, n_classes))
        super().__init__()
        self.reduce=Reduce('b n e -> b e',reduction='mean')
        self.layernorm=nn.LayerNorm(emb_size)
        self.linear=nn.Linear(emb_size,n_classes)
    def forward(self,x):
        x=self.reduce(x)
        feature=self.layernorm(x)
        x=self.linear(feature)
        return x,feature


class ViT(nn.Sequential):
    def __init__(self,
                patch_size: int = 10,
                emb_size: int = 10,
                img_size: int = 1000,
                depth: int = 6,#6
                n_classes: int = 3,
                **kwargs):
        super().__init__(

            PatchEmbedding( patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

