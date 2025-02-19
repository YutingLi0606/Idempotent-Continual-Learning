import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import copy
from timm.utils.model_ema import ModelEmaV2

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
# from models.utils.weight_interpolation_mobilenet import *
from models.utils.hessian_trace import hessian_trace
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    
    return parser


class DERidb(ContinualModel):
    NAME = 'deridb'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        #self.buffer2 = Buffer(self.args.buffer_size2, self.device)
        self.old_model = self.deepcopy_model(self.net)
        self.first_task=True
        self.s = backbone.num_classes
        self.c = 0
        self.net1=None
        self.ft=True
        self.task=0
        self.nets={}
       
        
    


    def observe(self, inputs, labels, not_aug_inputs):
        batch_size, _, H, W = inputs.shape
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
        x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
        targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
        x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
        inputs_sum = torch.cat([x1, x2], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        self.opt.zero_grad()
        outputs = self.net(inputs_sum)
        loss = self.loss(outputs, labels)
        if self.first_task:
            self.net1=self.deepcopy_model(self.net)
        else:
            self.net1=self.old_model
                          
        if not self.buffer.is_empty() and self.net1 is not None :
            
            if  not self.first_task:
                buf_inputs,buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
                B,_,H,W =buf_inputs.shape
                targets_buf1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
                x_buf1 = torch.cat([buf_inputs, targets_buf1 / self.s], dim=1)
                buf_logits1 = self.net(x_buf1)
                predictions = torch.nn.functional.softmax(buf_logits1, dim=1)
                pred_labels = predictions.argmax(dim=1).to(inputs.device)
                targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
                targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
                x_buf2 = torch.cat([buf_inputs, targets_buf2 / self.s], dim=1)
                buf_outputs=self.net1(x_buf2)
                
                loss += 0.3*  F.mse_loss(buf_outputs, buf_logits1)
                           
        loss.backward()
        self.opt.step()
        return loss.item()

   

    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)
       
        print('end_task call')

        
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
            for i, data in enumerate(dataset.train_loader):
                _, labels, not_aug_inputs = data
                not_aug_inputs = not_aug_inputs.to(self.device)
                self.buffer.add_data(examples=not_aug_inputs,labels=labels)
          

        else:
            self.old_model = self.deepcopy_model(self.net)
            self.buffer.empty()
            for i, data in enumerate(dataset.train_loader):
                _, labels, not_aug_inputs = data
                not_aug_inputs = not_aug_inputs.to(self.device)
                self.buffer.add_data(examples=not_aug_inputs,labels=labels)

        
        torch.save(self.old_model, 'old_model.pt')
        torch.save(self.net, 'net.pt')



    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        return model_copy
