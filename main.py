#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2022, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : main.py
@Time    : 2022/05/10 10:52:15
@Desc    : 
"""
import os
import torch
import pickle
import logging
import datetime
from flt import network
from torchvision import transforms
from argparse import ArgumentParser
from flt.algorithms import FedAvg, FedProx, MOON
from flt.utils.partitioner import IIDPartitioner, DirichletPartitioner
from flt.dataset import Cifar10Wrapper, Cifar100Wrapper, ImageFolderWrapper


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-b", "--backbone", default="SimpleCNN", type=str, choices=["SimpleCNN", "resnet34", "shufflenet"], 
                         help="the network/model for experiment")
    parser.add_argument("--net_config", default={}, type=lambda x: list(map(int, x.split(', '))), help="the federated learning network config")

    parser.add_argument("-d", "--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100", "mnist"],
                         help="the dataset for training Federated Learning")
    parser.add_argument("--alpha", default=0.05, type=float, help="the dirichlet ratio for dataset split to train Federated Learning")
    parser.add_argument("--datadir", default="./data/cifar10", type=str, help="the dataset dir")
    parser.add_argument('--partition', default="non-iid", type=str, choices=["iid", "non-iid"], help="the data partitioning strategy")

    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="the optimizer learning rate")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="the batch size for client local epoch training in federated learning")
    parser.add_argument("-wd", "--weight_decay", default=1e-5, type=float, help="the weight decay for optimizer in federated learning")
    parser.add_argument("-optim", "--optim_name", default="sgd", type=str, choices=["sgd", "adam", "amsgrad"],
                         help="the optimizer for client local epoch training in federated learning")
    parser.add_argument("-mu", "--mu", default=1.0, type=float, help="the mu for fedprox in federated learning")
    parser.add_argument("-n", "--n_parties", default=10, type=int, help="total client numbers in federated learning")
    parser.add_argument("-nk", "--nk_parties", default=10, type=int, help="client numbers for aggregation per communication round in federated learning")

    parser.add_argument("--epochs", default=3, type=int, help="the federated learning client local epoch for training")
    parser.add_argument("--rounds", default=50, type=int, help="the federated learning communication rounds")
    parser.add_argument("--alg", default="moon", type=str, choices=["fedavg", "fedprox", "moon"], help="the federated learning algorithm")

    parser.add_argument("--savedir", default="exps", type=str, help="the federated learning algorithm experiment save dir")
    return parser.parse_args()


def init_logger(savedir: str, filename: str):
    # ???????????????????????????
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    log_path = os.path.join(savedir, f"{filename}.log")
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def init_nets(backbone: str, n: int, net_config: dict = {}):
    """
    ??????????????????????????????
    :param str backbone: ?????????????????????, ??????????????????????????????
    :param int n: ?????????????????????
    :param dict net_config: ????????????????????????????????????????????????, defaults to {}
    :return dict: ?????????????????????
    """
    try:
        initalizer = getattr(network, backbone)
        clients = {}
        for idx in range(n):
            clients[idx] = initalizer(**net_config)
        return clients
    except AttributeError as e:
        logging.info(f"Network {backbone} can not found !!!")
        return 


def init_datasets(datadir: str, dataset: str, partition: str, n_parties: int, alpha: float = 0.5):
    """
    ??????????????????????????????????????????, ????????????????????????????????????
    :param str datadir: ????????????
    :param str dataset: ???????????????
    :param str partition: ?????????????????????
    :param int n_parties: ????????????????????????
    :param float alpha: dirichlet??????????????????, default for 0.5
    :return tuple: ?????????????????????, ???????????????
    """
    if dataset == "cifar10":
        datasets = Cifar10Wrapper(root=datadir, download=True, train=True)
        test_dataset = Cifar10Wrapper(root=datadir, download=True, train=False)
    elif dataset == "cifar100":
        datasets = Cifar100Wrapper(root=datadir, download=True, train=True)
        test_dataset = Cifar100Wrapper(root=datadir, download=True, train=False)
    elif dataset == "mnist":
        datasets = ImageFolderWrapper(root=datadir)
        test_dataset = ImageFolderWrapper(root=datadir)
    else:
        datasets = Cifar10Wrapper(root=datadir, download=True, train=True)
        test_dataset = Cifar10Wrapper(root=datadir, download=True, train=False)

    # ???????????????????????????????????????
    dataset_cache_dir = os.path.join("cache", dataset)
    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir)
    if partition == "iid":
        cache_filename = f"iid_{n_parties}.dataidx"
    else:
        cache_filename = f"non_iid_d_{alpha}_{n_parties}.dataidx"
    dataset_cache = os.path.join(dataset_cache_dir, cache_filename)
    if not os.path.exists(dataset_cache):
        logging.info(f"{cache_filename} file not exists, generate it")
        # ???????????????
        if partition == "iid":
            partitioner = IIDPartitioner(datasets, num=n_parties)
        elif partition == "non-iid":
            partitioner = DirichletPartitioner(datasets, num=n_parties, alpha=alpha)
        else:
            partitioner = IIDPartitioner(datasets, num=n_parties)
        dataidx_map = partitioner.partition()
        # ?????????????????????????????????
        with open(dataset_cache, "wb") as f:
            pickle.dump(dataidx_map, f)
    else:
        logging.info(f"{cache_filename} file exists, load it")
        with open(dataset_cache, "rb") as f:
            dataidx_map = pickle.load(f)
    dataidx_map_count = {k: len(v) for k, v in dataidx_map.items()}
    logging.info(f"dataidx for training {dataidx_map_count}")
    train_datasets, test_dataset = restruce_data_from_dataidx(datadir=datadir, dataset=dataset, dataidx_map=dataidx_map)
    cls_num_maps = {k: d.cls_num_map for (k, d) in train_datasets.items()}
    logging.info(cls_num_maps)
    return train_datasets, test_dataset


def restruce_data_from_dataidx(datadir: str, dataset: str, dataidx_map: dict):
    """
    ????????????dataidx?????????????????????????????????
    :param str datadir: ???????????????
    :param str dataset: ???????????????
    :param dict dataidx_map: ?????????????????????
    :return dict: ?????????????????????????????????
    """
    train_datasets = {}
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        for idx, dataidx in dataidx_map.items():
            train_datasets[idx] = Cifar10Wrapper(root=datadir, train=True, dataidxs=dataidx, download=False, transform=transform)
        test_dataset = Cifar10Wrapper(root=datadir, train=False, dataidxs=None, download=False, transform=transform)
    elif dataset == "cifar100":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            )
        ])
        for idx, dataidx in dataidx_map.items():
            train_datasets[idx] = Cifar100Wrapper(root=datadir, train=True, dataidxs=dataidx, download=False, transform=transform)
        test_dataset = Cifar100Wrapper(root=datadir, train=False, dataidxs=None, download=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        for idx, dataidx in dataidx_map.items():
            train_datasets[idx] = Cifar10Wrapper(root=datadir, train=True, dataidxs=dataidx, download=False, transform=transform)
        test_dataset = Cifar10Wrapper(root=datadir, train=False, dataidxs=None, download=False, transform=transform)
    return train_datasets, test_dataset


def init_algorithms(
    algorithm: str, global_net, nets: dict, train_datasets: dict, test_dataset, 
    nk_parties, E, comm_round, lr, batch_size, weight_decay, optim_name, device, savedir, *args, **kwargs):
    logging.info(f"Load {algorithm.upper()} for training")
    if algorithm == "fedavg":
        trainer = FedAvg(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            device=device, savedir=savedir
        )
    elif algorithm == "fedprox":
        trainer = FedProx(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, mu=kwargs.get("mu", 0.01),
            device=device, savedir=savedir
        )
    elif algorithm == "moon":
        trainer = MOON(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1), 
            temperature=kwargs.get("temperature", 0.5), 
            pool_size=kwargs.get("pool_size", 1), 
            device=device, savedir=savedir
        )
    else:
        trainer = None
    return trainer


def train(network: str, datadir: str, dataset: str, algorithm: str, partition: str, n_parties: int, alpha: float, savedir: str, args):
    if dataset in ["cifar10", "mnist"]:
        n_classes = 10
    elif dataset == "cifar100":
        n_classes = 100
    else:
        n_classes = 10
    # ?????????MOON???????????????????????????????????????????????????????????????
    if algorithm == "moon":
        net_config = {"model_name": f"{network}", "num_classes": n_classes}
        network = "ModelFedCon"
    else:
        net_config = {"num_classes": n_classes}
    # ????????????
    logging.info(f"Load network: {network}")
    global_nets = init_nets(network, 1, net_config)
    if global_nets is None or global_nets.get(0) is None:
        logging.info("Error, initialize global model failed")
        return 
    global_net = global_nets[0]
    nets = init_nets(network, n_parties, net_config)
    # ???????????????????????????????????????
    train_datasets, test_dataset = init_datasets(datadir, dataset, partition=partition, n_parties=n_parties, alpha=alpha)
    if nets is None or train_datasets is None or test_dataset is None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = init_algorithms(
        algorithm, global_net, nets, train_datasets, test_dataset, args.nk_parties, 
        args.epochs, args.rounds, args.learning_rate, args.batch_size, args.weight_decay,
        args.optim_name, device=device, savedir=savedir,
        # kwargs ??????
        mu=args.mu
    )
    if trainer is not None:
        trainer.start()
    else:
        logging.info("Trainer is None, please check it parameters")


if __name__ == "__main__":
    args = get_args()
    hash_name = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    savedir = os.path.join(args.savedir, hash_name)
    init_logger(savedir, hash_name)
    logging.info(args)
    train(
        network=args.backbone, datadir=args.datadir, dataset=args.dataset, 
        algorithm=args.alg, partition=args.partition, n_parties=args.n_parties, alpha=args.alpha, savedir=savedir, args=args
    )
