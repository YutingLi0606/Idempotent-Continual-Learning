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
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import copy
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer_bfp import Buffer
from torch.nn import functional as F
from datasets import get_dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.args import *
from torch.utils.data import DataLoader, TensorDataset

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser

def add_parser(parser):
	#parser.add_argument('--temperature', type=float, default=1,
	#			help="Weight of the influence of labels")


	parser.add_argument('--loss_type', type=str, default='mfro', choices=['mse', 'rmse', 'mfro', 'cos'],
				help='How to compute the matching loss on projected features.')
	parser.add_argument("--normalize_feat", action="store_true",
				help="if set, normalize features before computing the matching loss.")

	parser.add_argument("--proj_lr", type=float, default=0.1,
				help="Learning rate for the optimizer on the projectors.")  
	parser.add_argument("--momentum", type=float, default=0.9,
				help="Momentum for SGD.")
	#parser.add_argument("--id_type", type=str, default='js_div',
	#			help="type for loss.")
	parser.add_argument('--pool_dim', default='hw', type=str, choices=['h', 'w', 'c', 'hw', 'flatten'], 
				help="Pooling before computing BFP loss. If None, no pooling is applied.")			
	parser.add_argument("--class_balance", type=str2bool, default=False,
                        help="If set, the memory buffer will be balanced by class")
	return parser
def get_parser() -> ArgumentParser:
	parser = ArgumentParser(description='Continual learning via backward feature projection')
	add_management_args(parser)
	add_experiment_args(parser)
	add_rehearsal_args(parser)
    # model ensumble


	#parser.add_argument('--oldnet_type', type=str, default='lastnet', choices=['lastnet', 'maxmerge', 'ema'],
	#			help='How to merge to get the old net.')



				
	parser = add_parser(parser)
	
	return parser
def js_div(p, q):
    """Function that computes distance between two predictions"""
    p = p+1e-10
    q = q+1e-10
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(torch.log(p), m, reduction='batchmean') +
                  F.kl_div(torch.log(q), m, reduction='batchmean'))
def move_state_dict_to_device(state_dict, device):
    return {key: value.to(device) for key, value in state_dict.items()}
def merge_max_abs(theta_0, theta_1,theta_init, alpha ):
    """Mix multiple task vectors together by highest parameter value."""
    
        
    with torch.no_grad():
        
        task_vector_0 = {
        key: theta_0[key] -theta_init[key]
        for key in theta_0.keys()
    }
        task_vector_1 = {
        key: theta_1[key] -theta_init[key]
        for key in theta_1.keys()
    }
        
        new_task_vector = {
    key: torch.where(task_vector_1[key].abs() >= task_vector_0[key].abs(), task_vector_1[key], task_vector_0[key])
    for key in task_vector_0.keys()
}
        theta = {
        key: alpha * new_task_vector[key] + theta_init[key]
        for key in theta_init.keys()
    }
    return theta
def interpolate_weights(theta_0, theta_1, alpha ):
    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }
    return theta
def pool_feat(f1, f2, pool_dim, normalize_feat=False):
    assert f1.shape == f2.shape, (f1.shape, f2.shape) # (N, C, H, W)

    # If already pooled, return the original features
    if f1.ndim == 2:
        return f1, f2
    
    # Do the pooling and move the channel dim to the last
    if pool_dim == 'hw':
        f1 = f1.mean(dim=(2, 3)) # (N, C)
        f2 = f2.mean(dim=(2, 3)) # (N, C)
    elif pool_dim == 'c':
        f1 = f1.mean(dim=1).reshape(f1.shape[0], -1) # (N, H*W)
        f2 = f2.mean(dim=1).reshape(f2.shape[0], -1) # (N, H*W)
    elif pool_dim == 'h':
        f1 = f1.mean(dim=2).reshape(f1.shape[0], -1) # (N, C*W)
        f2 = f2.mean(dim=2).reshape(f2.shape[0], -1) # (N, C*W)
    elif pool_dim == 'w':
        f1 = f1.mean(dim=3).reshape(f1.shape[0], -1) # (N, C*H)
        f2 = f2.mean(dim=3).reshape(f2.shape[0], -1) # (N, C*H)
    elif pool_dim == 'flatten':
        f1 = f1.transpose(1, 3) # (N, W, H, C)
        f2 = f2.transpose(1, 3) # (N, W, H, C)
    else:
        raise ValueError("Unknown pooling dimension: {}".format(pool_dim))
        
    # Treat each example as an unit
    f1 = f1.reshape(f1.shape[0], -1) # (N, -1)
    f2 = f2.reshape(f2.shape[0], -1) # (N, -1)
    
    # Normalize features if needed
    if normalize_feat:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return f1, f2

def match_loss(f1, f2, loss_type):
    # Compute the loss according to the loss type
    if loss_type == 'mse':
        loss = F.mse_loss(f1, f2)
    elif loss_type == 'rmse':
        loss = F.mse_loss(f1, f2) ** 0.5
    elif loss_type == 'mfro':
        # Mean of Frobenius norm, normalized by the number of elements
        loss = torch.mean(torch.frobenius_norm(f1 - f2, dim=-1)) / (float(f1.shape[-1]) ** 0.5)
    elif loss_type == "cos":
        loss = 1 - F.cosine_similarity(f1, f2, dim=1).mean()
    else:
        raise ValueError("Unknown loss type: {}".format(loss_type))

    return loss

class Idempotent(ContinualModel):
    NAME = 'idempotent'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Idempotent, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device,class_balance = True)

        self.s = backbone.num_classes *self.args.temperature
        self.c = 0
        self.task=0
        self.class_num=0
        self.seen_to_last_task = torch.tensor([]).long().to(self.device)
        self.old_net = None
        self.first_task=True
        self.init_model=self.deepcopy_model(self.net)
 
 
        self.final_d = backbone.final_d
        self.bfp_flag = self.args.alpha_bfp > 0

        self.old_model = None
            
        
    


       

    
    def begin_task(self, dataset):
        
        self.class_num = dataset.N_CLASSES_PER_TASK
        
        
        if self.bfp_flag:
            print("Use BFP projector!")
            self.bfp_projector=nn.Linear(self.final_d,self.final_d)
            self.bfp_projector.to(self.device)
            self.opt_proj = SGD(
			self.bfp_projector.parameters(), 
			lr=self.args.proj_lr, momentum=self.args.momentum)
        else:
            print("Dont use BFP projector!")
    

    
    def compute_loss(self, feats, feats_old, mask_new, mask_old):
        bfp_loss = 0.0
        feat = feats[-1]
        feat_old = feats_old[-1]
			
			# After pooling, feat and feat_old have shape (n, d)
        feat, feat_old = pool_feat(feat, feat_old, self.args.pool_dim, self.args.normalize_feat)
        feat_proj = self.bfp_projector(feat) # (N, C)
        bfp_loss += self.args.alpha_bfp * match_loss(feat_proj, feat_old, self.args.loss_type)
        loss = bfp_loss
        loss_dict = {
			'match_loss': bfp_loss,
		}
        return loss, loss_dict
    
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



        outputs, feats = self.net.forward_all_layers(inputs_sum)
        



        loss_ce_current = self.loss(outputs, labels_sum)
        #print("loss_ce_current:",loss_ce_current)

       
        logits1  = self.net(x1)
        predictions_current = torch.nn.functional.softmax(logits1, dim=1)
        pred_labels = predictions_current.argmax(dim=1).to(inputs.device)
        targets_buf2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            
        x_2 = torch.cat([inputs, targets_buf2 / self.s], dim=1)
        outputs_id =self.net(x_2)
        if self.args.id_type=="logits":
            loss_id_current = F.mse_loss(outputs_id, logits1)
        if self.args.id_type=="softmax":
            predictions_current2 = torch.nn.functional.softmax(outputs_id, dim=1)
            loss_id_current = F.mse_loss(predictions_current, predictions_current2)
        if self.args.id_type=="js_div":
            predictions_current2 = torch.nn.functional.softmax(outputs_id, dim=1)
            loss_id_current = js_div(predictions_current, predictions_current2)
        #print("loss_id_current:",loss_id_current)




        
        


        loss_id_buffer = 0.0
        loss_ce_buffer = 0.0
        bfp_loss_all=0.0
        bfp_loss_dict = None

        if not self.buffer.is_empty() :
           
            buf_inputs, buf_labels,task_labels= self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
            B,_,H,W =buf_inputs.shape

            
            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            
            targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)

            x_bufs= torch.cat([xbuf1, xbuf2], dim=0)
            buf_labels_sum=torch.cat([buf_labels, buf_labels], dim=0)
            x_ouputs = self.net(x_bufs)
            loss_ce_buffer =  self.loss(x_ouputs,buf_labels_sum)
           # print("loss_ce_buffer:",loss_ce_buffer)
            

            
            if self.task > 0 :
                buf_inputs, buf_labels,task_labels= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
                
                #task_to_remove = self.task
                #mask = task_labels != task_to_remove
                #buf_inputs = buf_inputs[mask]
                #buf_labels = buf_labels[mask]
                
                B,_,H,W =buf_inputs.shape
                
                targets_buf1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
                x_buf1 = torch.cat([buf_inputs, targets_buf1 / self.s], dim=1)
                buf_logits1 = self.net(x_buf1)
                predictions_old = torch.nn.functional.softmax(buf_logits1, dim=1)
                pred_labels = predictions_old.argmax(dim=1).to(inputs.device)
                targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
                targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
                x_buf2 = torch.cat([buf_inputs, targets_buf2 / self.s], dim=1)
                buf_outputs=self.old_model(x_buf2)
 
                
                if self.args.id_type=="logits":
                    loss_id_buffer  =   F.mse_loss(buf_outputs, buf_logits1)
                if self.args.id_type=="softmax":
                    predictions_old2 = torch.nn.functional.softmax(buf_outputs, dim=1)
                    loss_id_buffer = F.mse_loss(predictions_old, predictions_old2)
                if self.args.id_type=="js_div":
                    predictions_old2 = torch.nn.functional.softmax(buf_outputs, dim=1)
                    loss_id_buffer = js_div(predictions_old, predictions_old2)
                #print("loss_id_buffer:",loss_id_buffer)
            

               
            
            if self.task > 0 and self.bfp_flag:
                buf_inputs, buf_labels,task_labels= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
                B,_,H,W =buf_inputs.shape

            
                targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
                xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            
                targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
                targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
                xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
                
                x_bufs= torch.cat([xbuf1, xbuf2], dim=0)

                buf_labels_sum=torch.cat([buf_labels, buf_labels], dim=0)
                inputs_comb = torch.cat((inputs_sum, x_bufs), dim=0)
                labels_comb = torch.cat((labels_sum, buf_labels_sum), dim=0)
                buf_logits_new_net, buf_feats_new_net = self.net.forward_all_layers(x_bufs)
                
                feats_comb = [torch.cat((f, bf), dim=0) for f, bf in zip(feats, buf_feats_new_net)]



                mask_old = labels_comb < self.task * self.class_num
                mask_new = labels_comb >= self.task * self.class_num

                    # Forward data through the old network to get the old features
                with torch.no_grad():
                    self.old_net.eval()
                    _, feats_old = self.old_net.forward_all_layers(inputs_comb)
                
                bfp_loss_all, bfp_loss_dict = self.compute_loss(
                    feats_comb, feats_old, mask_new, mask_old)
                #print("bfp_lo[ss_all:",bfp_loss_all)
            
				
        loss = loss_ce_current + self.args.weightb * loss_id_current + self.args.weightc * loss_ce_buffer + self.args.weighta * loss_id_buffer + bfp_loss_all

        if  self.bfp_flag : self.opt_proj.zero_grad()
        loss.backward()
        if  self.bfp_flag : self.opt_proj.step()
        
      
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:batch_size],task_labels=(torch.ones(self.args.batch_size) *self.task))

        return loss.item()
    
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)
        if self.first_task:
            self.first_task = False
            self.old_net = self.deepcopy_model(self.net)
            self.old_model = self.deepcopy_model(self.net)
        else:
            if self.args.oldnet_type == "lastnet":
                self.old_net = self.deepcopy_model(self.net)
                self.old_model = self.deepcopy_model(self.net)
            if self.args.oldnet_type == "maxmerge":
                
                theta_0 = self.old_model.state_dict()
                theta_1 = self.net.state_dict()
                theta_init=self.init_model.state_dict()
                theta_0 = move_state_dict_to_device(theta_0, self.device)
                theta_1 = move_state_dict_to_device(theta_1, self.device)
                theta_init = move_state_dict_to_device(theta_init, self.device)
                theta = merge_max_abs(theta_0, theta_1,theta_init, alpha =self.args.weightema )
                self.old_model.load_state_dict(theta, strict=True)
                self.old_model.to(self.device)
                buf_x, buf_y, buf_tl = self.buffer.get_all_data()
                dataset = TensorDataset(buf_x, buf_y)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                optimizer = optim.SGD(self.old_model.classifier.parameters(), lr=0.01, momentum=0.9)
                self.old_model.train()
                for epoch in range(30): 
                    for batch_inputs, batch_labels in dataloader:
                        optimizer.zero_grad() 
                        batch_size, _, H, W = batch_inputs.shape
                        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(self.device)
                        x1 = torch.cat([batch_inputs, targets_1 / self.s], dim=1)
                        targets_2 = torch.ones(batch_size, 1, H, W).to(self.device)
                        targets_2 = targets_2 * batch_labels.reshape(-1, 1, 1, 1) + 1
                        x2 = torch.cat([batch_inputs, targets_2 / self.s], dim=1)
                        inputs_sum = torch.cat([x1, x2], dim=0)
                        labels_sum = torch.cat([batch_labels, batch_labels], dim=0)
                        outputs = self.old_model(inputs_sum)  
                        loss = self.loss(outputs, labels_sum)  
                        loss.backward()  
                        optimizer.step()
                self.old_model.eval()
                print("classifier trained")
            if self.args.oldnet_type == "ema":
                theta_0 = self.old_model.state_dict()
                theta_1 = self.net.state_dict()
                theta = interpolate_weights(theta_0, theta_1, alpha=self.args.weightema )
                self.old_model.load_state_dict(theta, strict=True)
                self.old_model.to(self.device)
                buf_x, buf_y, buf_tl = self.buffer.get_all_data()
                dataset = TensorDataset(buf_x, buf_y)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                optimizer = optim.SGD(self.old_model.classifier.parameters(), lr=0.1, momentum=0.9)
                self.old_model.train()
                for epoch in range(30): 
                    for batch_inputs, batch_labels in dataloader:
                        optimizer.zero_grad() 
                        batch_size, _, H, W = batch_inputs.shape
                        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(self.device)
                        x1 = torch.cat([batch_inputs, targets_1 / self.s], dim=1)
                        targets_2 = torch.ones(batch_size, 1, H, W).to(self.device)
                        targets_2 = targets_2 * batch_labels.reshape(-1, 1, 1, 1) + 1
                        x2 = torch.cat([batch_inputs, targets_2 / self.s], dim=1)
                        inputs_sum = torch.cat([x1, x2], dim=0)
                        labels_sum = torch.cat([batch_labels, batch_labels], dim=0)
                        outputs = self.old_model(inputs_sum)  
                        loss = self.loss(outputs, labels_sum)  
                        loss.backward()  
                        optimizer.step()
                self.old_model.eval()
                theta_0 = self.old_model.state_dict()
                self.net.load_state_dict(theta_0, strict=True)
                print("classifier trained")

        
               
        print('end_task call')
    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy