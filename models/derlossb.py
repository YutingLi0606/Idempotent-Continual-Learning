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


class Derlossb(ContinualModel):
    NAME = 'derlossb'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Derlossb, self).__init__(backbone, loss, args, transform)
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
        mse = F.mse_loss(logits1, outputs_id, reduction='none')
        mse = mse.view(batch_size, -1).mean(dim=1)
        self.not_aug_inputs_all.append(not_aug_inputs.cpu()) 
        self.labels_all.append(labels.cpu())
        self.mse_all.append(mse.cpu())
        
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
            

            """
            targets_buf1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            x_buf1 = torch.cat([buf_inputs, targets_buf1 / self.s], dim=1)
            x_buf1 = torch.cat([x1, x_buf1], dim=0)
            buf_logits1 = self.net(x_buf1)
            predictions = torch.nn.functional.softmax(buf_logits1, dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            buf_inputs2 = torch.cat([inputs, buf_inputs], dim=0)
            B2,_,_,_ =buf_inputs2.shape
            targets_buf2 = torch.ones(B2, 1, H, W).to(inputs.device)
            targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            
            x_buf2 = torch.cat([buf_inputs2, targets_buf2 / self.s], dim=1)
            buf_outputs=self.net(x_buf2)
            
            loss += 1 *  F.mse_loss(buf_outputs, buf_logits1)
            """
            
            
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
            if self.task > 0:
                
                buf_x, buf_y, buf_tl = self.buffer.get_all_data()
                task_id = self.task-1
                mask = buf_tl == task_id
                buf_x = buf_x[mask]
                buf_y = buf_y[mask]
                buf_tl= buf_tl[mask]
                B,_,H,W =buf_x.shape
                targets_buf1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
                x_buf1 = torch.cat([buf_x, targets_buf1 / self.s], dim=1)
                buf_logits1 = self.net(x_buf1)
                predictions = torch.nn.functional.softmax(buf_logits1, dim=1)
                pred_labels = predictions.argmax(dim=1).to(inputs.device)
                targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
                targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
                x_buf2 = torch.cat([buf_x, targets_buf2 / self.s], dim=1)
                buf_outputs=self.net1(x_buf2)
                
                loss +=  0.3 * F.mse_loss(buf_outputs, buf_logits1) 
            """
            
                
                
   
            
            
            '''
            #targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            #targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            #xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            #buf_buf_inputs_sum = torch.cat([xbuf1,xbuf2], dim=0)
            buf_logits_sum = torch.cat([buf_logits, buf_logits2], dim=0)
            buf_inputs_sum = torch.cat([buf_inputs,buf_inputs], dim=0)
            #buf_outputs2 = self.net(xbuf2)
            #loss += 0.3 * F.mse_loss(buf_outputs2, buf_logits2)
            #buf_outputs=self.net(buf_buf_inputs_sum)
            #loss += 0.3 * F.mse_loss(buf_outputs, buf_logits_sum)
            predictions = torch.nn.functional.softmax(buf_logits_sum, dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf_sum = torch.ones(2*B, 1, H, W).to(inputs.device)
            targets_buf_sum = targets_buf_sum * pred_labels.reshape(-1, 1, 1, 1) + 1
            x_buf2 = torch.cat([buf_inputs_sum, targets_buf_sum / self.s], dim=1)
            buf_outputs_sum=self.net(x_buf2)
            
            
            loss += 0.5*F.mse_loss(buf_outputs_sum, buf_logits_sum)

            '''
            """
            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            inputs_buf = torch.cat([xbuf1, xbuf2], dim=0)
            labels_buf = torch.cat([buf_labels, buf_labels], dim=0)
            outputs_buf = self.net(inputs_buf)
            loss += self.loss(outputs_buf, labels_buf)d
            """
            
            '''
            predictions = torch.nn.functional.softmax(outputs_buf[:B], dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf = torch.ones(B, 1, H, W).to(inputs.device)
            targets_buf = targets_buf * pred_labels.reshape(-1, 1, 1, 1) + 1
            x_buf = torch.cat([buf_inputs, targets_buf / self.s], dim=1)
            buf_outputs_sum=self.net(x_buf)
            loss += 0.5*F.mse_loss(buf_outputs_sum, outputs_buf[:B])
            '''

            

        loss.backward()
      
        self.opt.step()
        #self.buffer.add_data(examples=not_aug_inputs, labels=labels[:batch_size],task_labels=(torch.ones(self.args.batch_size) *self.task))

        return loss.item()
    def end_epoch(self, dataset):
        not_aug_inputs_all = torch.cat(self.not_aug_inputs_all, dim=0).to(self.device) 
        labels_all = torch.cat(self.labels_all, dim=0).to(self.device)
        mse_all = torch.cat(self.mse_all, dim=0).to(self.device)
        examples_per_task = self.buffer.buffer_size // (self.task+1)
        if self.task==0:
            self.buffer.empty()
            topk_values, topk_indices = torch.topk(mse_all, examples_per_task, largest=False)
            selected_inputs = not_aug_inputs_all[topk_indices]
            selected_labels = labels_all[topk_indices]
            mse_all=mse_all[topk_indices]
            self.buffer.add_data(examples=selected_inputs, labels=selected_labels,task_labels=(torch.ones(examples_per_task) *self.task),mses=mse_all)
        
        if self.task > 0:
            # shrink buffer
            buf_x, buf_y, buf_tl,buf_mses = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                if ttl != self.task:
                    idx = (buf_tl == ttl)
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
            topk_values, topk_indices = torch.topk(mse_all, examples_per_task, largest=False)
            selected_inputs = not_aug_inputs_all[topk_indices]
            selected_labels = labels_all[topk_indices]
            mse_all=mse_all[topk_indices]
            self.buffer.add_data(examples=selected_inputs, labels=selected_labels,task_labels=(torch.ones(examples_per_task) *self.task),mses=mse_all)
        self.not_aug_inputs_all=[]
        self.labels_all=[]
        self.mse_all=[]
        print("data select!")
    
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
        else:
            self.old_model = self.deepcopy_model(self.net)
        """
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
                    batch_size, _, H, W = inputs.shape
                    targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
                    x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
                    outputs = self.net(x1, returnt='features')
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
        plt.savefig(f'tsne_visualization derloss task:{self.task}.png', dpi=300, bbox_inches='tight')
        plt.close()
        """



        
       
     
        
        print('end_task call')
    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy