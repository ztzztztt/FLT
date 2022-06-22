#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2021, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : cnn
@Time    : 2021/09/16 21:20
@Desc    :
"""
import torch
from torch import nn


def ConvBN(in_channel, out_channel, kernel, stride, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=(padding, padding), bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    x = torch.zeros((1, 3, 32, 32)).cuda()
    net = SimpleCNN().cuda()
    net(x)
