#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : moon.py
@Time    : 2022/06/17 14:57:58
@Desc    : moon算法的模型与普通模型不同, 需要单独定义
"""
import math
from torch import nn
from .simple_cnn import ConvBN
from .resnet import BasicBlock, BottleNeck
from .shufflenet import idenUnit, poolUnit


class ResNet_for_MOON(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = out.squeeze()
        y = self.fc(out)
        return out, out, y


def moon_resnet9(num_classes: int = 10):
    """ return a ResNet 9 object """
    return ResNet_for_MOON(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def moon_resnet18(num_classes: int = 10):
    """ return a ResNet 18 object """
    return ResNet_for_MOON(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def moon_resnet34(num_classes: int = 10):
    """ return a ResNet 34 object """
    return ResNet_for_MOON(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def moon_resnet50(num_classes: int = 10):
    """ return a ResNet 50 object """
    return ResNet_for_MOON(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def moon_resnet101(num_classes: int = 10):
    """ return a ResNet 101 object """
    return ResNet_for_MOON(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)

def moon_resnet152(num_classes: int = 10):
    """ return a ResNet 152 object """
    return ResNet_for_MOON(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)


class ShuffleNet_for_MOON(nn.Module):
    def __init__(self, output_size, scale_factor = 1, g = 8):
        super(ShuffleNet_for_MOON, self).__init__()
        self.g = g
        self.cs = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}

        c2 = self.cs[self.g]
        c2 = int(scale_factor * c2)
        c3, c4 = 2*c2, 4*c2

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size = 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        # build stages
        self.stage2 = self.build_stage(24, c2, repeat_time = 3, first_group = False, downsample = False)
        self.stage3 = self.build_stage(c2, c3, repeat_time = 7)
        self.stage4 = self.build_stage(c3, c4, repeat_time = 3)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(c4, output_size)
        self.weights_init()

    def build_stage(self, input_channel, output_channel, repeat_time, first_group = True, downsample = True):
        stage = [poolUnit(input_channel, output_channel, self.g, first_group = first_group, downsample = downsample)]
        
        for i in range(repeat_time):
            stage.append(idenUnit(output_channel, self.g)) # type: ignore

        return nn.Sequential(*stage) 

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        h = self.stage1(inputs)

        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)

        h = self.adaptive_pool(h, 1)
        h = h.squeze()
        y = self.fc(h)
        return h, h, y


def moon_shufflenet(num_classes: int = 10, g: int = 1, scale_factor: float = 0.5):
    return ShuffleNet_for_MOON(num_classes, g = g, scale_factor = scale_factor)


class SimpleCNN_for_MOON(nn.Module):
    def __init__(self, out_num: int = 10):
        super().__init__()
        self.block_1 = ConvBN(3, 16, 3, 1, 1)

        self.block_2 = nn.Sequential(
            ConvBN(16, 16, 1, 1),
            ConvBN(16, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_3 = nn.Sequential(
            ConvBN(16, 32, 1, 1),
            ConvBN(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_4 = nn.Sequential(
            ConvBN(32, 64, 1, 1),
            ConvBN(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block_5 = nn.Sequential(
            ConvBN(64, 128, 1, 1),
            ConvBN(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block_6 = nn.Sequential(
            ConvBN(128, 256, 1, 1),
            ConvBN(256, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_num)
        )

    def forward(self, x):
        h = self.block_1(x)
        h = self.block_2(h)
        h = self.block_3(h)
        h = self.block_4(h)
        h = self.block_5(h)
        h = self.block_6(h)
        h = h.squeeze()
        y = self.fc(h)
        return h, h, y


def moon_SimpleCNN(num_classes: int = 10):
    return SimpleCNN_for_MOON(out_num=num_classes)


class ModelFedCon(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        if model_name == "resnet9":
            basemodel = moon_resnet9(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet18":
            basemodel = moon_resnet18(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet34":
            basemodel = moon_resnet34(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet50":
            basemodel = moon_resnet50(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet101":
            basemodel = moon_resnet101(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet152":
            basemodel = moon_resnet152(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "shufflenet":
            basemodel = moon_shufflenet(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 288
        elif model_name == "SimpleCNN":
            basemodel = moon_SimpleCNN(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        else:
            basemodel = moon_SimpleCNN(num_classes=num_classes)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512

        self.l1 = nn.Linear(num_ftrs, num_ftrs)

        self.l2 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        y = self.l2(x)
        return h, x, y
