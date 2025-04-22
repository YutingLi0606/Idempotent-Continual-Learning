# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Erids(ContinualModel):
    NAME = 'erids'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Erids, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.c=0
        self.task=0
        self.s=backbone.num_classes

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        batch_size, _, H, W = inputs.shape
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
        x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
        targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
        x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
        inputs_sum = torch.cat([x1, x2], dim=0)
        labels_sum = torch.cat([labels, labels], dim=0)

        outputs = self.net(inputs_sum)
        loss = self.loss(outputs, labels_sum)
        loss.backward()
        self.opt.step()

        #self.buffer.add_data(examples=not_aug_inputs,
        #                    labels=labels[:real_batch_size])

        return loss.item()
    
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        not_aug_inputs_all=[]
        labels_all=[]
        mse_all=[]
        print(self.task)
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
        mean_value = torch.mean(mse_all)


        differences = torch.abs(mse_all - mean_value)

        topk_values, topk_indices = torch.topk(differences, 500, largest=True)
        selected_inputs = not_aug_inputs_all[topk_indices]
        selected_labels = labels_all[topk_indices]
        self.buffer.add_data(examples=selected_inputs,
                                 labels=selected_labels)

"""
    def end_task(self, dataset):
        print('\n\n')
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
    

