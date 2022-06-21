#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : __init__.py
@Time    : 2022/05/18 15:40:41
@Desc    : 
"""
from .simple_cnn import SimpleCNN
from .shufflenet import shufflenet
from .resnet import resnet9, resnet18, resnet34, resnet50, resnet101, resnet152
from .moon_net import moon_resnet9, moon_resnet18, moon_resnet34, moon_resnet50, \
    moon_resnet101, moon_resnet152, moon_shufflenet, moon_SimpleCNN, ModelFedCon