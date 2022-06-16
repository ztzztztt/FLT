#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : cifar10.py
@Time    : 2022/05/10 10:51:11
@Desc    : 
"""
import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from typing import Optional, List
from torchvision.datasets import CIFAR10


class Cifar10Wrapper(data.Dataset):
    def __init__(
            self, 
            root: str, 
            train: bool = True, 
            dataidxs: Optional[List[int]] = None, 
            download: bool = False, 
            transform = None, 
            target_transform = None
        ) -> None:
        self._root = root
        self._train = train
        self._dataidxs = dataidxs
        self._download = download
        self._transform = transform
        self._target_transform = target_transform
        self._datas, self._targets = self._build_datasets()

    def _build_datasets(self):
        cifar10_dataset = CIFAR10(
            root=self._root, 
            train=self._train, 
            transform=self._transform, 
            target_transform=self._target_transform, 
            download=self._download
        )
        if torchvision.__version__ == '0.2.1':
            if self._train:
                datas, targets = cifar10_dataset.train_data, np.array(cifar10_dataset.train_labels)   # type: ignore
            else:
                datas, targets = cifar10_dataset.test_data, np.array(cifar10_dataset.test_labels)     # type: ignore
        else:
            datas = cifar10_dataset.data
            targets = np.array(cifar10_dataset.targets)
        if self._dataidxs is not None:
            datas = datas[self._dataidxs]
            targets = targets[self._dataidxs]
        return datas, targets

    def __getitem__(self, index):
        data, target = self._datas[index], self._targets[index]
        data = Image.fromarray(data)
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target

    def __len__(self):
        if self._dataidxs is None:
            return self._targets.shape[0]
        else:
            return len(self._dataidxs)

    @property
    def datas(self):
        return self._datas
    
    @property
    def targets(self):
        return self._targets


if __name__ == "__main__":
    dataset = Cifar10Wrapper(
        root="data/cifar10",
        train=True,
        # dataidxs=[0, 1],
        download=False,
    )
    print(len(dataset))
    loader = data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=4
    )
    for (datas, targets) in loader:
        print(datas, targets)
