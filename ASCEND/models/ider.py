import torch
import copy
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import torch.nn.functional as F
from utils.args import *



def add_parser(parser):
    parser.add_argument('--weighta', type=float, help='Penalty weight for idempotence distillation.')
    parser.add_argument('--weightb', type=float, help='Penalty weight for current idempotence distillation.')
    parser.add_argument('--weightc', type=float, help='Penalty weight for er.')
    parser.add_argument('--weightmask', type=float, help='Penalty weight for mask ratio.')
    parser.add_argument("--class_balance", type=str2bool, default=True,
                        help="If set, the memory buffer will be balanced by class")		
    return parser
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Idempotent Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser = add_parser(parser)
    return parser

class Ider(ContinualModel):
    NAME = 'ider'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Ider, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device,class_balance = self.args.class_balance)
        self.ft=True
        self.task=0
        self.num_classs = backbone.num_classes
        self.s = backbone.num_classes
        self.first_task = True
        self.old_model=self.deepcopy_model(self.net)

        
    
    def observe(self, inputs, labels, not_aug_inputs):
        batch_size, _, H, W = inputs.shape

        self.opt.zero_grad()
        mask_current = torch.rand(1) > self.args.weightmask
        y_0_current = F.one_hot(labels, self.num_classs).float() if mask_current else torch.ones(batch_size, self.num_classs).to(self.device) /self.s
        z_current = self.net.f1(inputs)
        y_1_current, z1_current = self.net.f2(z_current, y_0_current)
        y_2_current, z2_current = self.net.f2(z_current , y_1_current.softmax(-1))
        loss_supervised_1 = self.loss(y_1_current, labels)
        loss_supervised_2 = self.loss(y_2_current, labels)

        loss = 0.5*(loss_supervised_1 + loss_supervised_2)
      
        if self.args.weightb!=0 and self.task>0:
            y_current_mask = torch.ones(batch_size, self.num_classs).to(self.device) /self.s
            z = self.net.f1(inputs)
            y_1, z1 = self.net.f2(z, y_current_mask)
            z_old = self.old_model.f1(inputs)
            y_2, z2 = self.old_model.f2(z_old, y_1.softmax(-1))
            loss += self.args.weightb*F.mse_loss(y_1, y_2)
      
        if not self.buffer.is_empty() and self.args.weightc !=0:
            buf_inputs,  buf_labels,_,_,_ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            batch_size, _, H, W = buf_inputs.shape
            mask = torch.rand(1) > self.args.weightmask
            y_0_buf = F.one_hot(buf_labels, self.num_classs).float() if mask else torch.ones(batch_size, self.num_classs).to(self.device) /self.s
            z_buf = self.net.f1(buf_inputs)
            y_1_buf, z1_buf = self.net.f2(z_buf, y_0_buf)
            y_2_buf, z2_buf = self.net.f2(z_buf , y_1_buf.softmax(-1))
            loss_supervised_1_buf = self.loss(y_1_buf, buf_labels)
            loss_supervised_2_buf = self.loss(y_2_buf, buf_labels)

            loss += self.args.weightc*(loss_supervised_1_buf + loss_supervised_2_buf)

        if not self.buffer.is_empty() and self.task>0 and self.args.weighta!=0:
            buf_inputs,  buf_labels,_,_,_ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            batch_size, _, H, W = buf_inputs.shape
           
            y_0 = torch.ones(batch_size, self.num_classs).to(self.device) /self.s
            
            z = self.net.f1(buf_inputs)
            y_1, z1 = self.net.f2(z, y_0)
         
            z_old = self.old_model.f1(buf_inputs)
            y_2, z2 = self.old_model.f2(z_old, y_1.softmax(-1))
            loss_unsupervised_y = F.mse_loss(y_1, y_2)
            loss += self.args.weighta * loss_unsupervised_y
     
 
        loss.backward()      
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels,logits=y_1_current.data, logits2=y_2_current.data,mask=y_0_current)

        return loss.item()
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)  

        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net).to(self.device)
        else:
            self.old_model = self.deepcopy_model(self.net).to(self.device)
            
    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        return model_copy
