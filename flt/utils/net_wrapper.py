#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : test.py
@Time    : 2022/05/24 16:38:41
@Desc    : 
"""
from torch import nn
from collections import OrderedDict


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers: dict, classifier_key: list = ["fc", "classifier", "l1"]):
        super(IntermediateLayerGetter, self).__init__(model.named_children())
        # 获取返回层不在网络中
        if not set(return_layers.values()).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        self._map_from_key = list(return_layers.values())
        self._map_to_key = list(return_layers.keys())

        # 获取特征提取网络、分类器模块的名称
        self._feature_layers = []
        self._fc_layers = []
        for name, _ in model.named_children():
            if name in classifier_key:
                self._fc_layers.append(name)
            else:
                self._feature_layers.append(name)
 
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            # 如果是feature extractor的最后一层，则需要将其展平
            if name == self._feature_layers[-1]:
                x = x.view(x.shape[0], -1)
            for idx, ok in enumerate(self._map_from_key):
                if ok == name:
                    out[self._map_to_key[idx]] = x
        return tuple(v for _, v in out.items())


Configs = {
    "ResNet":       {"h": "avg_pool",       "x": "avg_pool",        "y": "fc"},
    "ShuffleNet":   {"h": "adaptive_pool",  "x": "adaptive_pool",   "y": "fc"},
    "SimpleCNN":    {"h": "block_6",        "x": "block_6",         "y": "fc"},
}


def wrapper_net(net: nn.Module):
    net_name = net._get_name()
    if net_name not in Configs:
        raise ValueError(f"Wrapper Network Error, {net_name} config not exists")
    return IntermediateLayerGetter(net, Configs.get(net_name, {}))


if __name__ == "__main__":
    import torch
    from flt.network import resnet9, shufflenet, SimpleCNN
    nets = shufflenet()
    net = wrapper_net(nets)
    out = net(torch.rand(1, 3, 32, 32))
    h, x, y = out
    print(f"h: {h.shape}, x: {x.shape}, y: {y.shape}")
    # print([(k, v.shape) for k, v in out.items()])
    pass
