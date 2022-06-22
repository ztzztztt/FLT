#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : partition.py
@Time    : 2022/05/18 16:48:02
@Desc    : 
"""
import numpy as np
from decimal import Decimal


class Partitioner(object):
    def __init__(self, dataset, num: int) -> None:
        self._dataset = dataset
        self._num = num

    def partition(self):
        pass

    def count_target(self):
        # 获取当前的所有类别 id 以及每个类别的数量
        return np.unique(self._dataset.targets, return_index=False, return_counts=True)


class IIDPartitioner(Partitioner):
    def __init__(self, dataset, num: int) -> None:
        super().__init__(dataset, num)

    def partition(self):
        """
        数据切分
        :raises RuntimeError: 节点数量与实际生成的切片数量不一致抛出异常
        :return dict[int, list[int]]: 每个客户端切分后的数据样本下标
        """
        targets = self._dataset.targets
        # 获取当前的label，以及每个label的样本数量
        label, counts = self.count_target()
        # 类别的数量
        K = len(label)
        slices = {k: [] for k in range(self._num)}
        for class_idx in range(K):
            # 取出所有的当前的类别的样本下标
            sample_idxs = np.where(targets == class_idx)[0]
            np.random.shuffle(sample_idxs)
            class_counts = counts[class_idx]
            # 每份切分的数据样本长度
            step = (class_counts // self._num) if class_counts % self._num == 0 else (class_counts // self._num) + 1
            for i, start in enumerate(range(0, class_counts, step)):
                if i not in slices.keys():
                    raise RuntimeError("Client Id is not consistent with slice number !!!")
                slices[i].extend(sample_idxs[start: (start + step)])
        return slices


class DirichletPartitioner(Partitioner):
    def __init__(self, dataset, num: int, alpha: float = 0.5, min_require_size: int = 10) -> None:
        super().__init__(dataset, num)
        self._alpha = alpha
        self._min_require_size = min_require_size

    def partition(self):
        """
        数据切分
        :raises RuntimeError: 节点数量与实际生成的切片数量不一致抛出异常
        :return dict[int, list[int]]: 每个客户端切分后的数据样本下标
        """
        min_size = 0
        targets = self._dataset.targets
        # 获取当前的label，以及每个label的样本数量
        label, _ = self.count_target()
        # 类别的数量
        K = len(label)
        N = targets.shape[0]
        slices = {k: [] for k in range(self._num)}
        idx_batch = [[] for _ in range(self._num)]
        while min_size < self._min_require_size:
            idx_batch = [[] for _ in range(self._num)]
            for k in range(K):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                props = np.random.dirichlet(np.repeat(self._alpha, self._num))
                props = np.array([p * (len(idx_j) < N / self._num) for p, idx_j in zip(props, idx_batch)])
                props = props / props.sum()
                props = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, props))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(self._num):
            np.random.shuffle(idx_batch[j])
            slices[j] = idx_batch[j]
        return slices


class DeprecatedDirichletPartitioner(Partitioner):
    def __init__(self, dataset, num: int, alpha: float = 0.5) -> None:
        super().__init__(dataset, num)
        self._alpha = alpha

    def partition(self):
        """
        数据切分
        :raises RuntimeError: 节点数量与实际生成的切片数量不一致抛出异常
        :return dict[int, list[int]]: 每个客户端切分后的数据样本下标
        """
        targets = self._dataset.targets
        # 获取当前的label，以及每个label的样本数量
        label, counts = self.count_target()
        # 类别的数量
        K = len(label)
        slices = {k: [] for k in range(self._num)}
        probs = self._dirichlet(self._alpha, self._num, K)
        # 遍历每个类别
        for class_idx in range(K):
            # 取出所有的当前的类别的样本下标
            sample_idxs = np.where(targets == class_idx)[0]
            np.random.shuffle(sample_idxs)
            slice_number_lst = self._slice_num_lst(probs[class_idx], counts[class_idx])
            offset_lst = []
            for idx in range(len(slice_number_lst)):
                start = 0 if idx == 0 else sum(slice_number_lst[:idx])
                end = sum(slice_number_lst[:(idx+1)])
                offset_lst.append((start, end))
            # 每份切分的数据样本长度
            for i, (start, end) in enumerate(offset_lst):
                if i not in slices.keys():
                    raise RuntimeError("Client Id is not consistent with slice number !!!")
                slices[i].extend(sample_idxs[start:end])
        return slices

    def _slice_num_lst(self, prob: np.ndarray, count: int):
        """
        按照的每个类别的比例, 总数据量 将其设置为
        :param np.ndarray prob: 从狄利克雷分布抽样切分的比例
        :param int count: 当前样本总数
        :return list[int] : 返回的每个类别
        """
        counts_lst = (prob * count).tolist()
        counts_lst = [int(Decimal(count).quantize(Decimal("1."), rounding = "ROUND_HALF_UP")) for count in counts_lst]
        # 处理后的所有节点样本数量和超过所有的样本数量
        lst_count = sum(counts_lst)
        if lst_count > count:
            idx = np.argmax(counts_lst)
            counts_lst[idx] = counts_lst[idx] - (lst_count - count)
        elif lst_count < count:
            idx = np.argmin(counts_lst)
            counts_lst[idx] = counts_lst[idx] + (count - lst_count)
        return counts_lst

    def _dirichlet(self, alpha: float, client_num: int, classes_num: int):
        """
        生成采样概率
        :param float alpha: 狄利克雷分布采样浓度参数
        :param int client_num: 将数据划分成多少个拆分样本
        :param int classes_num: 一共有多少的类别数据
        :return np.ndarray: [classes_num, client_num]
        """
        return np.random.dirichlet(alpha * np.ones(shape=(client_num)), classes_num)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from flt.dataset import Cifar10Wrapper, Cifar100Wrapper
    dataset = Cifar10Wrapper(
        root="data/cifar10",
        train=True,
        download=False,
    )
    # partitioner = IIDPartitioner(dataset, 9)
    partitioner = DirichletPartitioner(dataset, 10, alpha=0.05)
    samples = partitioner.partition()
    sample_dict = {k:len(v) for k, v in samples.items()}
    print(sample_dict)
    # dataset_1 = Cifar10Wrapper(
    #     root="data/cifar10",
    #     train=True,
    #     dataidxs=r.get(0, []),
    #     download=False,
    # )
    # print(dataset_1)
    # print(len(dataset_1))
