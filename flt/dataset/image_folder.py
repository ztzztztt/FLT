#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : image_folder.py
@Time    : 2022/05/10 21:54:08
@Desc    : 
"""
import numpy as np
from torch.utils import data
from typing import Optional, List
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder, DatasetFolder


class ImageFolderWrapper(DatasetFolder):
    def __init__(
            self, 
            root: str, 
            dataidxs: Optional[List[int]] = None, 
            transform = None, 
            target_transform = None
        ) -> None:
        self._root = root
        self._dataidxs = dataidxs
        self._transform = transform
        self._target_transform = target_transform
        self._samples, self._loader = self._build_datasets()
    
    @property
    def cls_num_map(self):
        return {key: np.sum(key == self._targets) for key in np.unique(self._targets)}

    def _build_datasets(self):
        img_folder_dataset = ImageFolder(
            root=self._root, 
            transform=self._transform, 
            target_transform=self._target_transform, 
        )
        loader = img_folder_dataset.loader
        samples = img_folder_dataset.samples
        if self._dataidxs is not None:
            samples = np.array(samples)[self._dataidxs]
        else:
            samples = np.array(samples)
        return samples, loader

    def __getitem__(self, index: int):
        path, target = self._samples[index].tolist()
        data = self._loader(path)
        target = int(target)
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target

    def __len__(self):
        if self._dataidxs is None:
            return self._samples.shape[0]
        else:
            return len(self._dataidxs)

    @property
    def datas(self):
        datas = self._samples[:, 0]
        return datas
    
    @property
    def targets(self):
        targets = self._samples[:, 1].astype(np.int_)
        return targets


if __name__ == "__main__":
    dataset = ImageFolderWrapper(
        root="/home/chase/projects/FedSL/data/cifar10/d0.1/1",
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    print(len(dataset))
    print(dataset.datas)
    print(dataset.targets)
    loader = data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=4
    )
    for (datas, targets) in loader:
        print(datas, targets)
        break


