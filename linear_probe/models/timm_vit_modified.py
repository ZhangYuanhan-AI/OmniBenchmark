# ------------------------------------------------------------------------
# SenseTime VTAB
# Copyright (c) 2021 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
VTAB-SenseTime Model & Criterion Classes
"""
import timm

import torch
import copy
import torch.nn as nn
import torchvision
import os
from collections import OrderedDict


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                name = name.replace('fc_norm','norm')
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


class MyViT(nn.Module):
    def __init__(self, num_classes=1000, pretrain_path=None, enable_fc=False):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224',num_classes=1000)# pretrained=True)
        print('initializing ViT model as backbone using ckpt:', pretrain_path)
        state_dict = load_state_dict(pretrain_path)
        missing_keys,unexpected_keys = self.model.load_state_dict(state_dict,strict=False)
        print('missing_keys:', missing_keys)
        print('unexpected_keys:', unexpected_keys)

    def forward_features(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)

        return self.model.pre_logits(x[:, 0])


    def forward(self, x):
        features = self.forward_features(x)
        return features


def timm_vit_modified(**kwargs):

    default_kwargs={}
    default_kwargs.update(**kwargs)
    return MyViT(**default_kwargs)

def test_build():
    model = MyViT(init_ckpt='vit_base_patch32_224_in21k')
    image = torch.rand(2, 3, 224, 224)
    output = model(image)

if __name__ == '__main__':
    test_build()
