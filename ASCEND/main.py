# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import importlib
import os
import socket
import sys
import uuid
from argparse import ArgumentParser
import numpy  
import setproctitle
import torch
from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model
from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.continual_training import train as ctrain
from utils.distributed import make_dp
from utils.training import train
from utils.args import *

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser, args = parse_known_args()
    args = parse_model_args(parser, args)
    return args

def parse_known_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    return parser, args


def parse_model_args(parser, args):
    mod = importlib.import_module('models.' + args.model)

    def add_common_args(target_parser):
        target_parser.add_argument(
            "--savecheckpoint",
            type=str2bool,
            default=False,
            help="If set, save checkpoint after training",
        )

    add_common_args(parser)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        parser.add_argument('--half_data_in_first_task', action='store_true',
                            help='use half of data for first expirience')
        parser.add_argument('--device', type=str, default='cuda:0')

        has_buffer = hasattr(mod, 'Buffer')
        if has_buffer:
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')

        args = parser.parse_known_args()[0]
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]

        best = best[args.buffer_size] if has_buffer else best[-1]

        parser = getattr(mod, 'get_parser')()
        add_common_args(parser)

        to_parse = list(sys.argv[1:])
        while '--load_best_args' in to_parse:
            to_parse.remove('--load_best_args')
        to_parse += ['--' + k + '=' + str(v) for k, v in best.items()]

        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        return args

    parser = getattr(mod, 'get_parser')()
    add_common_args(parser)
    return parser.parse_args()


def run_experiment(args):
    if args.seed is not None:
        set_random_seed(args.seed)

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")
    

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)
    type(dataset).N_TASKS = args.n_tasks
    type(dataset).N_CLASSES_PER_TASK = type(dataset).N_CLASSES // type(dataset).N_TASKS
    print(args.model)
    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    print(args.n_epochs)
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()
    if args.model=="ider":  
        backbone = dataset.get_backboneid()
        print("use changed model middle")
    else:
        backbone = dataset.get_backbone()
    print(f'backbone number of parameters = {get_n_parameters(backbone)}')

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.distributed == 'dp':
        model.net = make_dp(model.net,args.device)
        model.to(args.device)
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


def get_n_parameters(backbone):
    p_count = 0
    for p in backbone.parameters():
        if p.requires_grad:
            p_count += p.nelement()
    return p_count


def main():
    lecun_fix()
    args = parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
