#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : fedavg.py
@Time    : 2022/05/19 21:12:39
@Desc    : MOON 联邦平均算法
"""
import os
import copy
import torch
import logging
import operator
import numpy as np
from torch import nn, optim
from torch.utils import data
from flt.utils.net_wrapper import wrapper_net


class MOON(object):
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
        mu: float,
        temperature: float,
        pool_size: int,
        *args, **kwargs
    ) -> None:
        """
        MOON 算法
        :param nn.Module global_net: 全局模型
        :param dict nets: 所有的局部模型
        :param dict datasets: 拆分的所有的数据集
        :param data.Dataset test_dataset: 测试数据集
        :param int nk_parties: 每轮选取多少个节点融合
        :param int E: 本地的epoch
        :param int comm_round: 通信的轮数
        :param float lr: 优化器学习率
        :param int batch_size: 优化器的batch大小
        :param float weight_decay: 优化器权重衰减系数
        :param str optim_name: 优化器的名称
        :param str device: 训练设备， GPU或者CPU
        :param str savedir: 模型保存路径
        :param str mu: 对比损失的权重
        :param str temperature: 对比损失计算权重
        :param str pool_size: 采用多大的旧模型池
        """
        self._global_net = wrapper_net(global_net)
        self._nets = {k: wrapper_net(net) for k, net in nets.items()}
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

        # MOON 算法相关，对比损失权重，对比相似度温度系数，使用多少个旧模型计算对比损失
        self._mu = mu
        self._temperature = temperature
        self._pool_size = pool_size

        self._args = args
        self._kwargs = kwargs

        # 设置全局模型不可训练
        self._global_net = self._require_grad_f(self._global_net)
        self._prev_nets = {}
        for key, net in self._nets.items():
            self._prev_nets[key] = [self._require_grad_f(copy.deepcopy(net))]

    def start(self):
        old_net_pool = []
        global_w = self._global_net.state_dict()
        for round in range(self._comm_round):
            logging.info(f"[Round] {round + 1} / {self._comm_round} start")
            # 选择部分或者全部节点进行训练
            samples = self._sample_nets(self._nets, self._nk_parties)
            net_w_lst, ratios = [], []
            for idx, (key, net) in enumerate(samples.items()):
                logging.info(f"  >>> [Local Train] client: {key} / [{idx + 1}/{len(samples)}]")
                net.load_state_dict(global_w)
                optimizer = self._optimizer(self._optim_name, net, lr=self._lr, weight_decay=self._weight_decay)
                net = self._train(
                    net, dataset=self._datasets[key], test_dataset=self._test_dataset, 
                    optimizer=optimizer, bs=self._bs, E=self._E, 
                    old_nets=self._prev_nets[key], mu=self._mu, temperature=self._temperature,
                    device=self._device
                )
                net_w_lst.append(net.state_dict())
                ratios.append(len(self._datasets[key]))
                # 保存所有的旧模型
                old_nets = self._prev_nets[key]
                if len(old_nets) < self._pool_size:
                    old_nets.append(self._require_grad_f(copy.deepcopy(net)))
                    self._prev_nets[key] = old_nets
                else:
                    for _ in range(len(old_nets) + 1 - self._pool_size):
                        del old_nets[0]
                    old_nets.append(self._require_grad_f(copy.deepcopy(net)))
                    self._prev_nets[key] = old_nets

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

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, old_nets: list, mu: float, temperature: float, device: torch.device):
        train_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        net = net.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        cosime = torch.nn.CosineSimilarity(dim=-1)
        for epoch in range(E):
            epoch_loss_lst = []
            epoch_loss1_lst = []
            epoch_loss2_lst = []
            net.train()
            for _, (x, y) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                _, prob, pred = net(x)
                loss1 = criterion(pred, y)

                # 计算全局模型的输出
                self._global_net.to(device)
                _, global_prob, _ = self._global_net(x)
                # self._global_net.to("cpu")

                positive = cosime(prob, global_prob)
                logits = positive.reshape(-1, 1)

                # 计算每一个旧模型的输出
                for old_net in old_nets:
                    old_net = old_net.to(device)
                    _, old_prob, _ = old_net(x)
                    negative = cosime(prob, old_prob)
                    logits = torch.cat([logits, negative.reshape(-1, 1)], dim=1)
                    # old_net.to("cpu")
                
                logits /= temperature
                labels = torch.zeros(x.size(0)).cuda().long()
                loss2 = mu * criterion(logits, labels)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()
            
                epoch_loss1_lst.append(loss1.item())
                epoch_loss2_lst.append(loss2.item())
                epoch_loss_lst.append(loss.item())
            
            logging.info(
                f"    >>> [Local Train] Epoch: {epoch + 1}, "
                f"Optim Loss: {(sum(epoch_loss1_lst) / len(epoch_loss1_lst)):.6f}, "
                f"Contrast Loss: {(sum(epoch_loss2_lst) / len(epoch_loss2_lst)):.6f}, "
                f"Total Loss: {(sum(epoch_loss_lst) / len(epoch_loss_lst)):.6f}"
            )

            with torch.no_grad():
                net.eval()
                correct, total = 0, 0
                for _, (x, y) in enumerate(test_dataloader):
                    x, y = x.to(device), y.to(device)
                    _, _, pred_y = net(x)
                    total += y.shape[0]
                    _, pred_label = torch.max(pred_y.data, 1)
                    correct += (pred_label == y.data).sum().item()
            logging.info(f"    >>> [Local Train] Epoch: {epoch + 1}, Acc: {(correct / total):.6f}")
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
                _, _, pred_y = net(x)
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

    def _require_grad_f(self, net):
        # 设置全局模型不可训练
        for param in net.parameters():
            param.requires_grad = False
        return net