# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import copy
from timm.utils.model_ema import ModelEmaV2

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
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


class Derlossh(ContinualModel):
    NAME = 'derlossh'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Derlossh, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.ft=True
        self.s = backbone.num_classes
        self.c = 0
        self.task=0
        self.old_model = self.deepcopy_model(self.net)
        self.first_task=True
        self.not_aug_inputs_all=[]
        self.labels_all=[]
        self.mse_all=[]

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        batch_size, _, H, W = inputs.shape
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
        x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
        targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
        x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
        inputs_sum = torch.cat([x1, x2], dim=0)
        labels_sum = torch.cat([labels, labels], dim=0)
       
 

        #outputs1 = self.net(x1)
    
        #outputs2 = self.net(x2)
        outputs = self.net(inputs_sum)
      
        loss = self.loss(outputs, labels_sum)

        
        logits1 = self.net(x1)
        predictions = torch.nn.functional.softmax(logits1, dim=1)
        pred_labels = predictions.argmax(dim=1).to(inputs.device)
        targets_buf2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            
        x_2 = torch.cat([inputs, targets_buf2 / self.s], dim=1)
        outputs_id=self.net(x_2)
            
        loss += self.args.weightb *  F.mse_loss(outputs_id, logits1)

        if self.first_task:
            self.net1=self.deepcopy_model(self.net)
        else:
            self.net1=self.old_model



        #loss = 1/2*(self.loss(outputs1, labels)+self.loss(outputs2, labels))
      

        if not self.buffer.is_empty() :
           
            buf_inputs, buf_labels,task_labels,_= self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
            B,_,H,W =buf_inputs.shape

            
            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            
            targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            x_bufs= torch.cat([xbuf1, xbuf2], dim=0)
            x_ouputs=self.net(x_bufs)
            buf_labels_sum=torch.cat([buf_labels, buf_labels], dim=0)
            loss +=  self.loss(x_ouputs,buf_labels_sum)
            #x_ouputs=self.net(xbuf1)
            #loss +=  self.loss(x_ouputs,buf_labels)
            


            
            if self.task > 0 :

                task_to_remove = self.task
                mask = task_labels != task_to_remove
                buf_inputs = buf_inputs[mask]
                buf_labels = buf_labels[mask]
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
            
                loss += self.args.weighta *  F.mse_loss(buf_outputs, buf_logits1)
            
            
            
            
            
            

            

        loss.backward()
      
        self.opt.step()
        #self.buffer.add_data(examples=not_aug_inputs, labels=labels[:batch_size],task_labels=(torch.ones(self.args.batch_size) *self.task))

        return loss.item()
    """
    def end_task(self, dataset):
        print('\n\n')
        print('\n\n')
        
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
        else:
            self.old_model = self.deepcopy_model(self.net)
        self.task+=1 
        self.build_buffer(dataset, self.task)
    def build_buffer(self, dataset, task):
        examples_per_task = self.buffer.buffer_size // task

        if task > 1:
            # shrink buffer
            buf_x, buf_y, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                idx = (buf_tl == ttl)
                ex, lab, tasklab = buf_x[idx], buf_y[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_task)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

        counter = 0
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                _, labels, not_aug_inputs = data
                not_aug_inputs = not_aug_inputs.to(self.device)
                if examples_per_task - counter > 0:

                    self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                         labels=labels[:(examples_per_task - counter)],
                                         task_labels=(torch.ones(self.args.batch_size) *
                                                      (task - 1))[:(examples_per_task - counter)])
                    counter += len(not_aug_inputs)
    """
    def end_task(self, dataset):
        print('\n\n')
        print(self.task)

        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
        else:
            self.old_model = self.deepcopy_model(self.net)
        not_aug_inputs_all=[]
        labels_all=[]
        mse_all=[]
        for i, data in enumerate(dataset.train_loader):
            inputs, labels, not_aug_inputs = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            not_aug_inputs = not_aug_inputs.to(self.device)
            with torch.no_grad():
                batch_size, _, H, W = inputs.shape
                targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
                x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
                logits1 = self.net(x1)
                predictions = torch.nn.functional.softmax(logits1, dim=1)
                pred_labels = predictions.argmax(dim=1).to(inputs.device)
                targets_buf2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
                targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            
                x_2 = torch.cat([inputs, targets_buf2 / self.s], dim=1)
                outputs_id=self.net(x_2)
                mse = F.mse_loss(logits1, outputs_id, reduction='none')
                mse = mse.view(batch_size, -1).mean(dim=1)
            not_aug_inputs_all.append(not_aug_inputs.cpu()) 
            labels_all.append(labels.cpu())
            mse_all.append(mse.cpu())

        not_aug_inputs_all = torch.cat(not_aug_inputs_all, dim=0).to(self.device) 
        labels_all = torch.cat(labels_all, dim=0).to(self.device)
        mse_all = torch.cat(mse_all, dim=0).to(self.device)
        #topk_values, topk_indices = torch.topk(mse_all, 500, largest=False)
        examples_per_task = self.buffer.buffer_size // ((self.task+1)*dataset.N_CLASSES_PER_TASK)
        if self.task==0:
            for label_class in labels_all.unique():
                idx = (labels_all == label_class)
                not_aug_inputs_class, lab, mse_class = not_aug_inputs_all[idx], labels_all[idx], mse_all[idx]

                topk_values, topk_indices = torch.topk(mse_class, examples_per_task, largest=False)
                selected_inputs = not_aug_inputs_class[topk_indices]
                selected_labels = lab[topk_indices]
                selected_mses=mse_class[topk_indices]
                unique_labels, counts = torch.unique(selected_labels, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    print(f"元素: {label.item()}, 数量: {count.item()}")
            
                self.buffer.add_data(examples=selected_inputs, labels=selected_labels,task_labels=(torch.ones(examples_per_task) *self.task),mses=selected_mses)
        
        if self.task > 0:
            # shrink buffer
            buf_x, buf_y, buf_tl,buf_mses = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_y.unique():
                idx = (buf_y == ttl)
                ex, lab, tasklab,buf_mse = buf_x[idx], buf_y[idx], buf_tl[idx],buf_mses[idx]
                first = min(ex.shape[0], examples_per_task)
                topk_values, topk_indices = torch.topk(buf_mse, first, largest=False)
                ex = ex[topk_indices]
                lab = lab[topk_indices]
                tasklab=tasklab[topk_indices]
                buf_mse=buf_mse[topk_indices]
                self.buffer.add_data(
                    examples=ex,
                    labels=lab,
                    task_labels=tasklab,
                    mses=buf_mse
                    )
            for label_class in labels_all.unique():
                idx = (labels_all == label_class)
                not_aug_inputs_class, lab, mse_class = not_aug_inputs_all[idx], labels_all[idx], mse_all[idx]

                topk_values, topk_indices = torch.topk(mse_class, examples_per_task, largest=False)
                selected_inputs = not_aug_inputs_class[topk_indices]
                selected_labels = lab[topk_indices]
                selected_mses=mse_class[topk_indices]
                unique_labels, counts = torch.unique(selected_labels, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    print(f"元素: {label.item()}, 数量: {count.item()}")
            
                self.buffer.add_data(examples=selected_inputs, labels=selected_labels,task_labels=(torch.ones(examples_per_task) *self.task),mses=selected_mses)
        self.not_aug_inputs_all=[]
        self.labels_all=[]
        self.mse_all=[]
        print("data select!")
       



        
       
     
        self.task+=1
        print(self.task)
        print('end_task call')
    
    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy