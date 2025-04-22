# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import copy
import torch
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, icarl_replay
from torch.utils.data import DataLoader, TensorDataset

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    return parser


def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_l, _ = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx]
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class],
                logits=_y_l[:samples_per_class],
                task_labels=(torch.ones(samples_per_class) *self.task)
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    norm_trans = dataset.get_normalization_transform()
    if norm_trans is None:
        def norm_trans(x): return x
    classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx + 1) * dataset.N_CLASSES_PER_TASK

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in (x, y, not_norm_x))
        a_x.append(not_norm_x.to('cpu'))
        a_y.append(y.to('cpu'))
        x_tarnsform=norm_trans(not_norm_x)
        batch_size, _, H, W = x_tarnsform.shape
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(x_tarnsform.device)
        x1 = torch.cat([x_tarnsform, targets_1 / self.s], dim=1)
        feats = self.net(x1, returnt='features')
        outs = self.net.classifier(feats)
        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)

    # 2.2 Compute class means
    for _y in a_y.unique():
        idx = (a_y == _y)
        _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
        feats = a_f[idx]
        mean_feat = feats.mean(0, keepdim=True)

        running_sum = torch.zeros_like(mean_feat)
        i = 0
        while i < samples_per_class and i < feats.shape[0]:
            cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

            idx_min = cost.argmin().item()

            mem_buffer.add_data(
                examples=_x[idx_min:idx_min + 1].to(self.device),
                labels=_y[idx_min:idx_min + 1].to(self.device),
                logits=_l[idx_min:idx_min + 1].to(self.device),
                task_labels=(torch.ones(1) *self.task)
            )

            running_sum += feats[idx_min:idx_min + 1]
            feats[idx_min] = feats[idx_min] + 1e6
            i += 1

    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size

    self.net.train(mode)


class ICarlid(ContinualModel):
    NAME = 'icarlid'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ICarlid, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.old_net = None
        self.task = 0
        self.c=0
        self.s=backbone.num_classes
        self.old_model = self.deepcopy_model(self.net)
        self.first_task=True
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.seen_to_last_task = torch.tensor([]).long().to(self.device)

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        
  
        feats = self.net(x, returnt='features')
        feats = feats.view(feats.size(0), -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, logits=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())
        present = labels.unique()
        
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        self.class_means = None
        batch_size, _, H, W = inputs.shape
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
        x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
        targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
        x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
        inputs_sum = torch.cat([x1, x2], dim=0)
        labels_sum = torch.cat([labels, labels], dim=0)
        
        if self.task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs_sum))
        self.opt.zero_grad()
        loss = self.get_loss(inputs_sum, labels_sum, self.task, logits)
        
        
        
        #self.opt.zero_grad()
        #outputs_sum = self.net(inputs_sum)
        #loss= self.get_loss2(inputs_sum,labels_sum,self.task)
        #loss = self.get_loss2(inputs_sum, labels_sum, self.task, logits)
        """
        in_class_seen = torch.isin(labels, self.seen_to_last_task)

        
        labels_new = labels[~in_class_seen]
        inputs_new = inputs[~in_class_seen]
        B_new, _, H, W = inputs_new.shape
        if B_new>0:
        
            targets_1 = self.c * torch.ones(B_new, 1, H, W).to(inputs.device)
            x1 = torch.cat([inputs_new, targets_1 / self.s], dim=1)

            logits1 = self.net(x1)
            predictions = torch.nn.functional.softmax(logits1, dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf2 = torch.ones(B_new, 1, H, W).to(inputs.device)
            targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            
            x_2 = torch.cat([inputs_new, targets_buf2 / self.s], dim=1)
            outputs_id=self.net(x_2)
            
            loss += self.args.weightb *  F.mse_loss(outputs_id, logits1)
        
        if self.first_task:
            self.net1=self.deepcopy_model(self.net)
        else:
            self.net1=self.old_model
        labels_old = labels[in_class_seen]
        inputs_old = inputs[in_class_seen]
        B_old, _, H, W = inputs_old.shape
        if B_old>0:
            targets_1 = self.c * torch.ones(B_old, 1, H, W).to(inputs.device)
            x1 = torch.cat([inputs_old, targets_1 / self.s], dim=1)

            logits1 = self.net(x1)
            predictions = torch.nn.functional.softmax(logits1, dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf2 = torch.ones(B_old, 1, H, W).to(inputs.device)
            targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            
            x_2 = torch.cat([inputs_old, targets_buf2 / self.s], dim=1)
            outputs_id=self.net1(x_2)
            
            loss += self.args.weighta *  F.mse_loss(outputs_id, logits1)
        """
        



        #loss = 1/2*(self.loss(outputs1, labels)+self.loss(outputs2, labels))
      
        """
        if not self.buffer.is_empty() :
            
            buf_inputs, buf_labels,_,task_labels= self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
            B,_,H,W =buf_inputs.shape

        
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
        """
        
        loss.backward()

        self.opt.step()

        return loss.item()
        

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs)[:, :ac]
        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        return loss
    def get_loss2(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs)[:, pc:ac]
       
            # Compute loss on the current task
        targets = self.eye[labels][:, pc:ac]
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        assert loss >= 0

        return loss
    
    def begin_task(self, dataset):
        icarl_replay(self, dataset)
    """
    def end_epoch(self, dataset) -> None:
        if self.task>0:
            examples, labels, _ ,_= self.buffer.get_all_data()
            dataset = TensorDataset(examples, labels)
            batch_size = 32
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for inputs, labels in train_loader:
                batch_size, _, H, W = inputs.shape
                targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(self.device)
                x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
                targets_2 = torch.ones(batch_size, 1, H, W).to(self.device)
                targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
                x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
                inputs_sum = torch.cat([x1, x2], dim=0)
                labels_sum = torch.cat([labels, labels], dim=0)
        
            if self.task > 0:
                with torch.no_grad():
                    logits = torch.sigmoid(self.old_net(inputs_sum))
            self.opt.zero_grad()
            loss = self.get_loss(inputs_sum, labels_sum, self.task, logits)
            loss.backward()

            self.opt.step()
    """



    def end_task(self, dataset) -> None:
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
        else:
            self.old_model = self.deepcopy_model(self.net)
        self.seen_to_last_task=self.seen_so_far
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            fill_buffer(self, self.buffer, dataset, self.task)
        self.task += 1
        self.class_means = None

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _ ,_= self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                allt = None
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]
                    batch_size, _, H, W = batch.shape
                    targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(batch.device)
                    x1 = torch.cat([batch , targets_1 / self.s], dim=1)
                    feats = self.net(x1, returnt='features').mean(0)
                    if allt is None:
                        allt = feats
                    else:
                        allt += feats
                        allt /= 2
                class_means.append(allt.flatten())
        self.class_means = torch.stack(class_means)
    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy