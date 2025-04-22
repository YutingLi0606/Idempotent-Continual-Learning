import copy
import math
import torch


import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F

from utils.buffer_bfp import Buffer
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import MNISTMLP
from utils.args import *
from utils.lowrank_reg import LowRankReg
from utils.routines import forward_loader_all_layers

#from .projector_manager import ProjectorManager
def add_parser(parser):
	#parser.add_argument('--alpha_bfp', type=float, default=1,
	#			help="Weight of the backward feature projection loss. It can be overridden by the 'alpha_bfpX' below")
	parser.add_argument('--alpha_bfp1', type=float, default=None)
	parser.add_argument('--alpha_bfp2', type=float, default=None)
	parser.add_argument('--alpha_bfp3', type=float, default=None)
	parser.add_argument('--alpha_bfp4', type=float, default=None)

	parser.add_argument('--loss_type', type=str, default='mfro', choices=['mse', 'rmse', 'mfro', 'cos'],
				help='How to compute the matching loss on projected features.')
	parser.add_argument("--normalize_feat", action="store_true",
				help="if set, normalize features before computing the matching loss.")
	parser.add_argument("--opt_type", type=str, default="sgdm", choices=["sgd", "sgdm", "adam"],
				help="Optimizer type.")
	parser.add_argument("--proj_lr", type=float, default=0.1,
				help="Learning rate for the optimizer on the projectors.")    
	parser.add_argument("--momentum", type=float, default=0.9,
				help="Momentum for SGD.")

	parser.add_argument('--proj_init_identity', action="store_true",
				help="If set, initialize the projectors to the identity mapping.")
	parser.add_argument('--proj_task_reset', type=str2bool, default=True,
				help="If set, initialize the projectors to a random mapping.")

	parser.add_argument('--proj_type', type=str, default="1", choices=['0', '1', '2', '0p+1'],
				help="Type of the backward feature projection. (number of layers in MLP projector)")
	parser.add_argument('--final_feat', action='store_true',default=True,
				help="If true, bfp loss will only be applied to the last feature map.")
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

	parser.add_argument("--base_method", type=str, default='derpp', choices=['derpp', 'er'], 
				help="Base method to use, determining the default hyperparameters")

    # alpha in the original paper
	parser.add_argument('--alpha_distill', type=float, default=None,
				help='Weight of the replayed distillation loss.')
    # beta in the original paper
	parser.add_argument('--alpha_ce', type=float, default=None,
				help='Weight of the replayed CE loss.')

	parser.add_argument('--resnet_skip_relu', action="store_true",
				help="If set, the last ReLU of each block of ResNet is skipped.")

	parser.add_argument("--no_old_net", action="store_true",
				help="If set, the behavior will be like the old network is not available.")
	parser.add_argument("--use_buf_logits", action="store_true",
				help="If set, logits distillation will use the logits replayed from the buffer")
	parser.add_argument('--use_buf_feats', action="store_true",
				help="If set, BFP will use the features replayed from the buffer")

	# On which data the BFP loss is applied
	parser.add_argument("--old_only", action="store_true",
				help="If set, the BFP loss will be applied on the buffered data. ")
	parser.add_argument("--new_only", action="store_true",
				help="If set, the BFP loss will be applied on the online data. ")

	parser.add_argument("--no_resample", action="store_true",
				help="If set, the replayed data will not be resampled for each loss.")
    
				
	parser = add_parser(parser)
	
	return parser

BEST_ARGS_DERPP = {
	'seq-cifar10': {
		200: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.2, 
			"alpha_ce": 0.5, "n_epochs": 50},
	},
	'seq-cifar100': {
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
		2000: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
	},
	'seq-tinyimg': {
		4000: {"lr": 0.1, "minibatch_size": 64, "batch_size": 64, "alpha_distill": 0.3, 
			"alpha_ce": 0.8, "n_epochs": 100},
	},
}

BEST_ARGS_ER = {
	'seq-cifar10': {
		200: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
		500: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
	},
	'seq-cifar100': {
		500: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
		2000: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
	},
	'seq-tinyimg': {
		4000: {"lr": 0.1, "minibatch_size": 64, "batch_size": 64, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 100},
	},
}

BEST_ARGS = {
	'derpp': BEST_ARGS_DERPP,
	'er': BEST_ARGS_ER,
}


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
class ProjectorManager(nn.Module):
	'''
	Helper class managing the projection layers for BFP
	Such that it can be easily integrated into other continual learning methods
	'''
	def __init__(self, args, net_channels, device):
		super(ProjectorManager, self).__init__()
		self.args = args
		self.net_channels = net_channels
		self.device = device
		
		# Initialize the backward projection layers
		self.alpha_bfp_list = [self.args.alpha_bfp] * len(self.net_channels)
		if self.args.alpha_bfp1 is not None: self.alpha_bfp_list[0] = self.args.alpha_bfp1
		if self.args.alpha_bfp2 is not None: self.alpha_bfp_list[1] = self.args.alpha_bfp2
		if self.args.alpha_bfp3 is not None: self.alpha_bfp_list[2] = self.args.alpha_bfp3
		if self.args.alpha_bfp4 is not None: self.alpha_bfp_list[3] = self.args.alpha_bfp4
		self.bfp_flag = sum(self.alpha_bfp_list) > 0
		
		self.reset_proj()
		
		# Get the list of layers where BFP is applied
		if self.args.final_feat:
			self.layers_bfp = [-1]
		else:
			self.layers_bfp = list(range(len(self.net_channels)))

	def begin_task(self, dataset) :
		if not self.bfp_flag: return

		if self.args.proj_task_reset:
			self.reset_proj()

	def _get_projector(self, feat_dim, init_identity=False):
		if self.args.proj_type == '0':
			projector = nn.Identity()
		elif self.args.proj_type == "1":
			projector = nn.Linear(feat_dim, feat_dim)
			if init_identity:
				projector.weight.data = torch.eye(feat_dim)
				projector.bias.data = torch.zeros(feat_dim)
		elif self.args.proj_type == "2":
			projector = nn.Sequential(
				nn.Linear(feat_dim, feat_dim),
				nn.ReLU(),
				nn.Linear(feat_dim, feat_dim),
			)
		else:
			raise Exception("Unknown projector type: {}".format(self.args.proj_type))

		projector.to(self.device)
		return projector
		
	def reset_proj(self):
		# Get one optimizer for each network layer
		self.projectors = nn.ModuleList()
		for c in self.net_channels:
			projector = self._get_projector(c, self.args.proj_init_identity)
			self.projectors.append(projector)

		if self.args.proj_type != '0':
			# Optimizer for all projectors
			if self.args.opt_type == 'sgd':
				self.opt_proj = SGD(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr)
			elif self.args.opt_type == 'sgdm':
				self.opt_proj = SGD(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr, momentum=self.args.momentum)
			elif self.args.opt_type == 'adam':
				self.opt_proj = Adam(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr)
		else:
			self.opt_proj = None

	def compute_loss(self, feats, feats_old, mask_new, mask_old):
		bfp_loss = 0.0

		for i in self.layers_bfp:
			projector = self.projectors[i]
			feat = feats[i]
			feat_old = feats_old[i]
			
			# After pooling, feat and feat_old have shape (n, d)
			feat, feat_old = pool_feat(feat, feat_old, self.args.pool_dim, self.args.normalize_feat)
			
			feat_proj = projector(feat) # (N, C)
			
			bfp_loss += self.alpha_bfp_list[i] * match_loss(feat_proj, feat_old, self.args.loss_type)

		bfp_loss /= len(self.layers_bfp)

		loss = bfp_loss

		loss_dict = {
			'match_loss': bfp_loss,
		}

		return loss, loss_dict

	def before_backward(self):
		if not self.bfp_flag: return
		if self.opt_proj is not None: self.opt_proj.zero_grad()

	def end_task(self, dataset, net):
		if not self.bfp_flag: return

	def step(self):
		if not self.bfp_flag: return
		if self.opt_proj is not None: self.opt_proj.step()
class Bfpid(ContinualModel):
    NAME = 'bfpid'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        
        
        # Then call the super constructor
        super(Bfpid, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.task_id=0
        self.s = backbone.num_classes
        self.class_num=0
        self.c= 0
        self.net1= copy.deepcopy(self.net)
        self.buffer = Buffer(self.args.buffer_size, self.device, class_balance = self.args.class_balance)
        
        print(self.args.alpha_distill)
        print(self.args.alpha_ce)
        print(self.args.use_buf_feats)
        
        # if resnet_skip_relu, modify the backbone to skip relu at the end of each major block
        if self.args.resnet_skip_relu:
            self.net.skip_relu(last=self.args.final_feat)

        # For domain-IL MNIST datasets, we should use the logits from the buffer
        if self.args.dataset in ['perm-mnist', 'rot-mnist']:
            self.args.use_buf_logits = True

        # if the old net is not used, then set the old_only and use_buf_logits flags
        if self.args.no_old_net:
            self.args.old_only = True
            self.args.use_buf_logits = True

        assert not (self.args.new_only and self.args.old_only)
        
        # initialize the projectors used for BFP
        self.projector_manager = ProjectorManager(self.args, self.net.net_channels, self.device)

    def begin_task(self, dataset):
        self.class_num = dataset.N_CLASSES_PER_TASK
        self.projector_manager.begin_task(dataset)
        print(self.projector_manager.bfp_flag)

    def observe(self, inputs, labels, not_aug_inputs):
        # Regular CE loss on the online data
        
        batch_size, _, H, W = inputs.shape
        batch_size_inputs=batch_size
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
        x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
        targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
        x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
        inputs_sum = torch.cat([x1, x2], dim=0)
        labels_sum = torch.cat([labels, labels], dim=0)

        outputs, feats = self.net.forward_all_layers(inputs_sum)
        ce_loss = self.loss(outputs, labels_sum)
        """
        logits1,_ = self.net.forward_all_layers(x1)
        predictions = torch.nn.functional.softmax(logits1, dim=1)
        pred_labels = predictions.argmax(dim=1).to(inputs.device)
        targets_buf2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            
        x_2 = torch.cat([inputs, targets_buf2 / self.s], dim=1)
        outputs_id,_=self.net.forward_all_layers(x_2)
            
        ce_loss += self.args.weightb *  F.mse_loss(outputs_id, logits1)
		"""
		
        
        def sample_buffer_and_forward(transform = self.transform):
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=transform)
            buf_inputs, buf_labels, buf_logits, buf_task_labels,buf_logits2 = buf_data[0], buf_data[1], buf_data[2], buf_data[3],buf_data[4]
            buf_feats = [buf_data[5]] if self.args.use_buf_feats else None
            buf_feats2 = [buf_data[6]] if self.args.use_buf_feats else None
            batch_size, _, H, W = buf_inputs.shape
            targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
            x1_buf = torch.cat([buf_inputs, targets_1 / self.s], dim=1)
            targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
            targets_2 = targets_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            x2_buf = torch.cat([buf_inputs, targets_2 / self.s], dim=1)
            buf_inputs_sum = torch.cat([x1_buf, x2_buf], dim=0)
            buf_labels_sum = torch.cat([buf_labels, buf_labels], dim=0)
            buf_logits_sum = torch.cat([buf_logits, buf_logits2], dim=0)
            buf_task_labels_sum = torch.cat([buf_task_labels,buf_task_labels], dim=0)
            buf_feats_sum = torch.cat([buf_feats, buf_feats2], dim=0)if self.args.use_buf_feats else None
            buf_logits_new_net, buf_feats_new_net = self.net.forward_all_layers(buf_inputs_sum)

            return buf_inputs_sum, buf_labels_sum, buf_logits_sum, buf_task_labels_sum, buf_feats_sum, buf_logits_new_net, buf_feats_new_net
       

        logits_distill_loss = 0.0
        replay_ce_loss = 0.0
        bfp_loss_all = 0.0
        bfp_loss_dict = None

        if not self.buffer.is_empty():
            '''Distill loss on the replayed images'''
            
            if self.args.alpha_distill > 0:
                if self.args.no_resample and "buf_inputs" in locals(): pass # No need to resample
                else: buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()
       
                if (not self.args.use_buf_logits) and (self.old_net is not None):
                    with torch.no_grad():
                        buf_logits = self.old_net(buf_inputs)
                        
                logits_distill_loss = self.args.alpha_distill * F.mse_loss(buf_logits_new_net, buf_logits)
                
                """
                else:
                    if (not self.args.use_buf_logits) and (self.old_net is not None) and self.task_id>0:
                          
                        buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                        buf_inputs, buf_labels, buf_logits, buf_task_labels,buf_logits2 = buf_data[0], buf_data[1], buf_data[2], buf_data[3],buf_data[4]
                        B,_,H,W =buf_inputs.shape
                
                        targets_buf1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
                        x_buf1 = torch.cat([buf_inputs, targets_buf1 / self.s], dim=1)
                        buf_logits1,_ = self.net.forward_all_layers(x_buf1)
                        predictions = torch.nn.functional.softmax(buf_logits1, dim=1)
                        pred_labels = predictions.argmax(dim=1).to(inputs.device)
                        targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
                        targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
                        x_buf2 = torch.cat([buf_inputs, targets_buf2 / self.s], dim=1)
                        buf_outputs,_=self.net1.forward_all_layers(x_buf2)
                        
                        logits_distill_loss = self.args.weighta *  F.mse_loss(buf_outputs, buf_logits1)
				
                """
                
                
                
                
                
                  

            '''CE loss on the replayed images'''
            if self.args.alpha_ce > 0:
                """
                if self.args.no_resample and "buf_inputs" in locals(): pass # No need to resample
                else: buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()
                
                #replay_ce_loss = self.args.alpha_ce * self.loss(buf_logits_new_net, buf_labels)
                replay_ce_loss = self.loss(buf_logits_new_net, buf_labels)
                """
                buf_inputs, buf_labels,_,buf_task_labels,_= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
                B,_,H,W =buf_inputs.shape

            
                targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
                xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            
                targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
                targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
                xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
                x_bufs= torch.cat([xbuf1, xbuf2], dim=0)
                x_ouputs,_=self.net.forward_all_layers(x_bufs)
                buf_labels_sum=torch.cat([buf_labels, buf_labels], dim=0)
                replay_ce_loss =  self.loss(x_ouputs,buf_labels_sum)
            else:
                  print("no replay ce loss!")

            '''Backward feature projection loss'''
            if self.old_net is not None and self.projector_manager.bfp_flag:
                if not self.args.new_only:
                    buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()

                if self.args.use_buf_feats:
                    # new and old features should be both a list 
                    # And in this case, we only care about the last layer
                    feats_comb = buf_feats_new_net[-1:]
                    feats_old = buf_feats
                    mask_new = torch.ones_like(buf_labels).bool()
                    mask_old = torch.zeros_like(buf_labels).bool()
                else:
                    # Inputs, feats and labels for the online and buffer data, concatenated
                    if self.args.new_only:
                        inputs_comb = inputs_sum
                        labels_comb = labels_sum
                        feats_comb = feats
                    elif self.args.old_only:
                        mask_old = buf_labels < self.task_id * self.class_num
                        inputs_comb = buf_inputs[mask_old]
                        labels_comb = buf_labels[mask_old]
                        feats_comb = [f[mask_old] for f in  buf_feats_new_net]
                    else:
                        inputs_comb = torch.cat((inputs_sum, buf_inputs), dim=0)
                        labels_comb = torch.cat((labels_sum, buf_labels), dim=0)
                        feats_comb = [torch.cat((f, bf), dim=0) for f, bf in zip(feats, buf_feats_new_net)]

                    mask_old = labels_comb < self.task_id * self.class_num
                    mask_new = labels_comb >= self.task_id * self.class_num

                    # Forward data through the old network to get the old features
                    with torch.no_grad():
                        self.old_net.eval()
                        _, feats_old = self.old_net.forward_all_layers(inputs_comb)
                
                bfp_loss_all, bfp_loss_dict = self.projector_manager.compute_loss(
                    feats_comb, feats_old, mask_new, mask_old)
        self.opt.zero_grad()      
        #loss = ce_loss + logits_distill_loss + replay_ce_loss + bfp_loss_all+ loss_id
        loss = ce_loss + logits_distill_loss + replay_ce_loss + bfp_loss_all
        

        #self.opt.zero_grad()
        self.projector_manager.before_backward()

        loss.backward()
        
        self.opt.step()
        self.projector_manager.step()

        task_labels = torch.ones_like(labels) * self.task_id
        if self.args.use_buf_feats:
            # Store the unpooled version of the final-layer features in the buffer
            final_feats = feats[-1]
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs[:batch_size_inputs,:].data, 
                                task_labels=task_labels,
                                logits2=outputs[batch_size_inputs:,:].data,
                                final_feats=final_feats[:batch_size_inputs,:].data,
                                final_feats2=final_feats[batch_size_inputs:,:].data           )
        else:
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs[:batch_size_inputs,:].data, 
                                task_labels=task_labels,
                                logits2=outputs[batch_size_inputs:,:].data)

        
        return loss.item()

    def end_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)
        self.net1 = copy.deepcopy(self.net)
        self.task_id+=1
        self.old_net.eval()

        self.projector_manager.end_task(dataset, self.old_net)