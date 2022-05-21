#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : fedavg.py
@Time    : 2022/05/19 21:12:39
@Desc    : FedAVG 联邦平均算法
"""
import os
import torch
import logging
import operator
import numpy as np
from torch import nn, optim
from torch.utils import data


class FedAvg(object):
    def __init__(
        self,
        global_net,
        nets: dict,
        datasets: dict,
        test_dataset: data.Dataset,
        nk_parties: int,
        E: int,
        comm_round: int,
        lr: float,
        batch_size: int,
        weight_decay: float,
        optim_name: str,
        device: str,
        savedir: str,
        *args, **kwargs
    ) -> None:
        self._global_net = global_net
        self._nets = nets
        self._datasets = datasets
        self._test_dataset = test_dataset
        self._nk_parties = nk_parties
        self._E = E
        self._comm_round = comm_round
        
        self._lr = lr
        self._bs = batch_size
        self._weight_decay = weight_decay
        self._optim_name = optim_name

        self._device = torch.device(f"cuda") if device == "cuda" else torch.device("cpu")

        self._savedir = savedir
        self._args = args
        self._kwargs = kwargs

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, device: torch.device):
        train_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        net = net.to(device)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(E):
            epoch_loss_lst = []
            net.train()
            for _, (x, y) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred_y = net(x)
                loss = criterion(pred_y, y)
                loss.backward()
                optimizer.step()
                epoch_loss_lst.append(loss.item())
            

            with torch.no_grad():
                net.eval()
                correct, total = 0, 0
                for _, (x, y) in enumerate(test_dataloader):
                    x, y = x.to(device), y.to(device)
                    pred_y = net(x)
                    total += y.shape[0]
                    _, pred_label = torch.max(pred_y.data, 1)
                    correct += (pred_label == y.data).sum().item()
            logging.info(f"    >>> [Local Train] Epoch: {epoch + 1}, Loss: {sum(epoch_loss_lst) / len(epoch_loss_lst)}, Acc: {correct / total}")
        return net

    def _aggregate(self, net_w_lst: list, ratios: list):
        sample_num = sum(ratios)
        global_w = net_w_lst[0]
        for key in global_w.keys():
            if "num_batches_tracked" not in key:
                global_w[key] *= (ratios[0] / sample_num)
        for key in global_w.keys():
            for i in range(1, len(net_w_lst)):
                if "num_batches_tracked" not in key:
                    global_w[key] += (ratios[i] / sample_num) * net_w_lst[i][key]        
        return global_w

    def _valid(self, net, dataset, bs, device):
        net = net.to(device)
        test_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=False, drop_last=True)
        with torch.no_grad():
            net.eval()
            correct, total = 0, 0
            for _, (x, y) in enumerate(test_dataloader):
                x, y = x.to(device), y.to(device)
                pred_y = net(x)
                total += y.shape[0]
                _, pred_label = torch.max(pred_y.data, 1)
                correct += (pred_label == y.data).sum().item()
            return correct / total

    def _optimizer(self, optim_name, net, lr, weight_decay: float = 1e-5):
        if optim_name == "sgd":
            return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optim_name == "adam":
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        elif optim_name == "amsgrad":
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)

    def _sample_nets(self, nets, nk_parties):
        if nk_parties > len(nets):
            logging.info(f"Error, can not sample {nk_parties} client over {len(nets)} clients")
            return nets
        else:
            idxs = [idx for idx in range(len(nets))]
            sampled = np.random.choice(idxs, size=nk_parties, replace=False)
            samples = {}
            for idx in sampled:
                samples[idx] = nets[idx]
            samples = dict(sorted(samples.items(), key=operator.itemgetter(0)))
            return samples

    def start(self):
        global_w = self._global_net.state_dict()
        for round in range(self._comm_round):
            logging.info(f"[Round] {round + 1} / {self._comm_round} start")
            # 选择部分或者全部节点进行训练
            samples = self._sample_nets(self._nets, self._nk_parties)
            net_w_lst, ratios = [], []
            for idx, (key, net) in enumerate(samples.items()):
                logging.info(f"  >>> [Local Train] client: {key} / [{idx + 1}/{len(samples)}]")
                net.load_state_dict(global_w)
                optimizer = self._optimizer(self._optim_name, net, lr=self._lr, weight_decay=1e-5)
                net = self._train(
                    net, dataset=self._datasets[key], test_dataset=self._test_dataset, 
                    optimizer=optimizer, bs=self._bs, E=self._E, device=self._device
                )
                net_w_lst.append(net.state_dict())
                ratios.append(len(self._datasets[key]))
            # 模型聚合
            global_w = self._aggregate(net_w_lst, ratios)
            self._global_net.load_state_dict(global_w)
            acc = self._valid(self._global_net, self._test_dataset, self._bs, self._device)
            # 保存模型
            logging.info(f"[Gloabl] Round: {round + 1}, Acc: {acc}")
            if not os.path.exists(f"{self._savedir}/models/"):
                os.makedirs(f"{self._savedir}/models/")
            torch.save(
                self._global_net.state_dict(), f"{self._savedir}/models/global_round_{round+1}.pth"
            )
