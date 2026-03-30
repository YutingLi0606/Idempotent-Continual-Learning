# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torchvision import transforms
from copy import deepcopy

def icarl_replay(self, train_loader, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """
        
    if self.task_id > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(train_loader)
        
        data_concatenate = torch.cat if type(train_loader.dataset.data) == torch.Tensor else np.concatenate
        need_aug = hasattr(train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            refold_transform = lambda x: x.cpu()
        else:    
            data_shape = len(train_loader.dataset.data[0].shape)
            if data_shape == 3:
                refold_transform = lambda x: (x.cpu()*255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                refold_transform = lambda x: (x.cpu()*255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        train_loader.dataset.targets = np.concatenate([
            train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
            ])
        train_loader.dataset.data = data_concatenate([
            train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
            ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
                ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
                ])

class Buffer(nn.Module):
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, num_seen_examples=0, blocked=False, n_tasks=None, mode='reservoir', class_balance=False):
        super().__init__()
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        
        self.class_balance = class_balance
        if self.class_balance: print("Using class balanced buffer")
        else: print("Using standard reservoir buffer")

        self.num_seen_examples = num_seen_examples
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'grad_inputs', "final_feats","logits2","final_feats2","mask"]

        self.blocked = blocked

    def get_extra_state(self):
        state = {
            'num_seen_examples': self.num_seen_examples,
            'blocked': self.blocked
        }
        return state

    def set_extra_state(self, state):
        self.num_seen_examples = state['num_seen_examples']
        self.blocked = state['blocked']

    def turn_on_blocking(self):
        self.blocked = True
    
    def turn_off_blocking(self):
        self.blocked = False

    def reservoir(self, num_seen_examples: int, buffer_size: int) -> int:
        """
        Reservoir sampling algorithm.
        :param num_seen_examples: the number of seen examples
        :param buffer_size: the maximum buffer size
        :return: the target index if the current image is sampled, else -1
        """
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1

    def reservoir_balanced(self, num_seen_examples:int, buffer_size:int) -> int:
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            # Return a random index corresponding to classes that have most examples in the buffer
            y = self.labels
            classes, counts = torch.unique(y, return_counts = True)
            max_count = counts.max()
            classes_max = classes[counts == max_count]
            idx_max = torch.stack([y==c for c in classes_max], dim=0).float().sum(0)
            idx_max = (idx_max > 0).nonzero().squeeze()
            idx = np.random.choice(idx_max.cpu().numpy())
            
            return idx
        else:
            return -1


    def ring(self, num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
        return num_seen_examples % buffer_portion_size + task * buffer_portion_size

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor, 
                     grad_inputs:torch.Tensor, final_feats:torch.Tensor,logits2: torch.Tensor,final_feats2:torch.Tensor,mask:torch.Tensor,
                     ) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                tensor = torch.zeros((self.buffer_size,
                    *attr.shape[1:]), dtype=typ, device=self.device)
                # setattr(self, attr_str, tensor)
                self.register_buffer(attr_str, tensor)

    def get_index_reservoir(self, examples):
        '''
        Get the insertion index of the given examples, using reservoir sampling
        :param examples: the examples to insert
        :return: the insertion index
        '''
        index = []
        for i in range(len(examples)):
            if self.class_balance:
                idx = self.reservoir_balanced(self.num_seen_examples, self.buffer_size)
            else:
                idx = self.reservoir(self.num_seen_examples, self.buffer_size)
            index.append(idx)
            self.num_seen_examples += 1
        return index

    def add_data(self, examples, indices=None, 
                 labels=None, logits=None, task_labels=None, grad_inputs=None, final_feats=None,logits2=None,final_feats2=None,mask=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param indices: indices at which the examples should be inserted
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param grad_inputs: tensor containing the gradients of output w.r.t. of the inputs
        :return:
        """
        if self.blocked:
            return
        
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, grad_inputs, final_feats, logits2,final_feats2,mask)

        for i in range(examples.shape[0]):
            if indices is not None:
                index = indices[i]
            elif self.class_balance:
                index = self.reservoir_balanced(self.num_seen_examples, self.buffer_size)
            else:
                index = self.reservoir(self.num_seen_examples, self.buffer_size)

            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if grad_inputs is not None:
                    self.grad_inputs[index] = grad_inputs[i].to(self.device)
                if final_feats is not None:
                    self.final_feats[index] = final_feats[i].to(self.device)
                if logits2 is not None:
                    self.logits2[index] = logits2[i].to(self.device)
                if final_feats2 is not None:
                    self.final_feats2[index] = final_feats2[i].to(self.device)
                if mask is not None:
                    self.mask[index] = mask[i].to(self.device)


    def get_data(self, size: int, transform: transforms=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        
        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0 or not hasattr(self, "examples"):
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
