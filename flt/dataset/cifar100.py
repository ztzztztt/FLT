#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : cifar100.py
@Time    : 2022/05/10 21:51:37
@Desc    : 
"""
import torchvision
import numpy as np
from typing import Optional, List
from torch.utils import data
from torchvision.datasets import CIFAR100


class Cifar100Wrapper(data.Dataset):
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
    
    @property
    def cls_num_map(self):
        return {key: np.sum(key == self._targets) for key in np.unique(self._targets)}

    def _build_datasets(self):
        cifar100_dataset = CIFAR100(
            root=self._root, 
            train=self._train, 
            transform=self._transform, 
            target_transform=self._target_transform, 
            download=self._download
        )
        if torchvision.__version__ == '0.2.1':
            if self._train:
                datas, targets = cifar100_dataset.train_data, np.array(cifar100_dataset.train_labels)   # type: ignore
            else:
                datas, targets = cifar100_dataset.test_data, np.array(cifar100_dataset.test_labels)     # type: ignore
        else:
            datas = cifar100_dataset.data
            targets = np.array(cifar100_dataset.targets)
        if self._dataidxs is not None:
            datas = datas[self._dataidxs]
            targets = targets[self._dataidxs]
        return datas, targets
    
    def __getitem__(self, index):
        data, target = self._datas[index], self._targets[index]
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
    dataset = Cifar100Wrapper(
        root="data/cifar100",
        train=True,
        download=False
    )
    print(len(dataset))
    loader = data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=4
    )
    for (datas, targets) in loader:
        print(datas, targets)

