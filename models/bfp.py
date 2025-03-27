import copy
import math
import torch
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F

from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import MNISTMLP
from utils.args import *
#from utils.lowrank_reg import LowRankReg
#rom utils.routines import forward_loader_all_layers


from .utils import *
#from .projector_manager import ProjectorManager
def add_parser(parser):
	parser.add_argument('--alpha_bfp', type=float, required=True,
				help="Weight of the backward feature projection loss. It can be overridden by the 'alpha_bfpX' below")
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
	parser.add_argument('--final_feat', action='store_true',
				help="If true, bfp loss will only be applied to the last feature map.")
	parser.add_argument('--pool_dim', default='hw', type=str, choices=['h', 'w', 'c', 'hw', 'flatten'], 
				help="Pooling before computing BFP loss. If None, no pooling is applied.")			
	
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
class Bfp(ContinualModel):
    NAME = 'bfp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        

        # Then call the super constructor
        super(Bfp, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.task_id=0
        self.buffer = Buffer(self.args.buffer_size, self.device, class_balance = self.args.class_balance)

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
        self.projector_manager.begin_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs):
        # Regular CE loss on the online data
        outputs, feats = self.net.forward_all_layers(inputs)
        ce_loss = self.loss(outputs, labels)

        def sample_buffer_and_forward(transform = self.transform):
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=transform)
            buf_inputs, buf_labels, buf_logits, buf_task_labels = buf_data[0], buf_data[1], buf_data[2], buf_data[3]
            buf_feats = [buf_data[4]] if self.args.use_buf_feats else None
            buf_logits_new_net, buf_feats_new_net = self.net.forward_all_layers(buf_inputs)

            return buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net

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

            '''CE loss on the replayed images'''
            if self.args.alpha_ce > 0:
                if self.args.no_resample and "buf_inputs" in locals(): pass # No need to resample
                else: buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()
                
                replay_ce_loss = self.args.alpha_ce * self.loss(buf_logits_new_net, buf_labels)

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
                        inputs_comb = inputs
                        labels_comb = labels
                        feats_comb = feats
                    elif self.args.old_only:
                        mask_old = buf_labels < self.task_id * self.args.N_CLASSES_PER_TASK
                        inputs_comb = buf_inputs[mask_old]
                        labels_comb = buf_labels[mask_old]
                        feats_comb = [f[mask_old] for f in  buf_feats_new_net]
                    else:
                        inputs_comb = torch.cat((inputs, buf_inputs), dim=0)
                        labels_comb = torch.cat((labels, buf_labels), dim=0)
                        feats_comb = [torch.cat((f, bf), dim=0) for f, bf in zip(feats, buf_feats_new_net)]

                    mask_old = labels_comb < self.task_id * self.args.N_CLASSES_PER_TASK
                    mask_new = labels_comb >= self.task_id * self.args.N_CLASSES_PER_TASK

                    # Forward data through the old network to get the old features
                    with torch.no_grad():
                        self.old_net.eval()
                        _, feats_old = self.old_net.forward_all_layers(inputs_comb)
                
                bfp_loss_all, bfp_loss_dict = self.projector_manager.compute_loss(
                    feats_comb, feats_old, mask_new, mask_old)
                
        loss = ce_loss + logits_distill_loss + replay_ce_loss + bfp_loss_all

        self.opt.zero_grad()
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
                                logits=outputs.data, 
                                task_labels=task_labels,
                                final_feats=final_feats.data)
        else:
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs.data, 
                                task_labels=task_labels)

        log_dict = {
            "train/loss": loss, 
            "train/ce_loss": ce_loss, 
            "train/logits_distill_loss": logits_distill_loss,
            "train/replay_ce_loss": replay_ce_loss,
            "train/bfp_loss_all": bfp_loss_all,
        }
        if bfp_loss_dict is not None:
            for k, v in bfp_loss_dict.items(): log_dict.update({"train/" + k: v})
        wandb.log(log_dict)

        return loss.item()

    def end_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)
        self.task_id+=1
        self.old_net.eval()

        self.projector_manager.end_task(dataset, self.old_net)