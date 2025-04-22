# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
import torch
import torch.utils.data
from torch.optim import SGD, Adam
import torch.optim as optim
import copy
from timm.utils.model_ema import ModelEmaV2
from datasets import get_dataset
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from torch.nn import functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.ft=True
        self.task=0
        #self.final_d = backbone.final_d
        #self.bfp_projector = nn.Linear(self.final_d,self.final_d).to(self.device)
        



    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy
    '''
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)
        
        self.net.eval()
        features=[]
        labels_sum=[]
        for k, test_loader in enumerate(dataset.test_loaders):
            if k>2:
                break
            for data in test_loader:
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs, returnt='features')
                    features.append(outputs.cpu().numpy()) 
                    labels_sum.append(labels.cpu().numpy())
        features = np.concatenate(features)
        labels_sum = np.concatenate(labels_sum)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features)
        print("t-SNE complete!")
        plt.figure(figsize=(10, 8))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        colors = [
           '#FF0000',  # 红
           '#00FF00',  # 绿
           '#0000FF',  # 蓝
           '#FFD700',  # 金
           '#FF1493',  # 粉
           '#00FFFF',  # 青
           '#FF8C00',  # 橙
           '#8A2BE2',  # 紫
           '#32CD32',  # 绿
           '#FF69B4'   # 粉红
           ]
        for i, class_name in enumerate(class_names):
            idx = labels_sum == i
            plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], 
                       c=[colors[i]], label=class_name, alpha=0.6)
        plt.title('t-SNE Visualization of CIFAR-10')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'tsne_visualization der task:{self.task}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('end_task call')
        '''
        
