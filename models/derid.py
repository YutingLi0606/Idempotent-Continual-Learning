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
def interpolate_weights(theta_0, theta_1, alpha):
    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }
    return theta




class DERid(ContinualModel):
    NAME = 'derid'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.old_model = self.deepcopy_model(self.net)
        self.first_task=True
        self.s = 100
        self.c = 0
        self.net1=None
        self.ft=True
        self.task=0
        self.nets={}
        self.ema_model = ModelEmaV2(
            self.deepcopy_model(self.net),
            decay=0.6,
            device=self.device)
       
        
    
  
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
        
        
        
        
        '''
        predictions = torch.nn.functional.softmax(outputs_0, dim=1)
        pred_labels = predictions.argmax(dim=1).to(inputs.device)
        targets2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets2 = targets2 * pred_labels.reshape(-1, 1, 1, 1) + 1
        x_2 = torch.cat([inputs, targets2 / self.s], dim=1)
        outputs_logits=self.net(x_2)
        loss += 0.01*F.mse_loss(outputs_0, outputs_logits)

        '''
       
                
            
               
               
        
        if not self.buffer.is_empty() and self.net1 is not None :
            
            if  not self.first_task:
                buf_inputs, buf_labels,buf_logits,buf_logits2,task_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
                B,_,H,W =buf_inputs.shape
                task_to_remove = self.task
                mask = task_labels != task_to_remove
                buf_inputs = buf_inputs[mask]
                buf_labels = buf_labels[mask]
                buf_logits = buf_logits[mask]
                task_labels= task_labels[mask]
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
                
                task_labels +=1
                max_task = task_labels.max()
                coefficients_re = (max_task - task_labels) / max_task
                coefficients_id = (task_labels) / max_task
                coefficients_re = coefficients_re.view(B,1)
                coefficients_id = coefficients_id.view(B,1)
                loss_re = 0.3 * F.mse_loss(buf_logits1, buf_logits, reduction='none')
                loss_id = 0.3 * F.mse_loss(buf_outputs, buf_logits1,reduction='none')
                B_logits,N = loss_re.shape
                #loss += 0.3*  F.mse_loss(buf_outputs, buf_logits1)
              
                
                weighted_loss_re = loss_re * coefficients_re.expand(-1, N)
                weighted_loss_id = loss_id * coefficients_id.expand(-1, N)
                loss += weighted_loss_re.mean()+ weighted_loss_id.mean()
               
                #loss += 0.1 * F.mse_loss(buf_logits1, buf_logits)
                '''
                if self.task- 4 > 0:
                    mask = task_labels < (self.task - 4)
                    buf_inputs_re = buf_inputs[mask]
                    buf_logits_re = buf_logits[mask]
                    B_re,_,H,W =buf_inputs_re.shape
                    targets_buf1 = self.c * torch.ones(B_re, 1, H, W).to(inputs.device)
                    x_buf1_re = torch.cat([buf_inputs_re, targets_buf1 / self.s], dim=1)
                    if B_re>0:
                        buf_logits1_re = self.net(x_buf1_re)
                        loss += 0.3 * F.mse_loss(buf_logits1_re, buf_logits_re)
                '''
            
           
                #loss += 0.3 * F.mse_loss(buf_logits1, buf_logits)
            #buf_logits_old= self.net1(x_buf1)
            #loss += 0.3*F.mse_loss(buf_logits1, buf_logits_old)
           
            
            
            #logits1 = torch.nn.functional.log_softmax(logits1, dim=-1)
            #logits2 = torch.nn.functional.softmax(logits2, dim=-1)
            #loss = torch.nn.functional.kl_div(logits1, logits2, reduction='batchmean')
            
           
            '''
            predictions_logits = torch.nn.functional.softmax(buf_logits, dim=1)
            pred_labels_logits = predictions_logits.argmax(dim=1).to(inputs.device)
            targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
            targets_buf2 = targets_buf2 * pred_labels_logits.reshape(-1, 1, 1, 1) + 1
            x_buf2 = torch.cat([buf_inputs, targets_buf2 / self.s], dim=1)
            buf_outputs=self.net(x_buf2)
            loss +=  0.5*F.mse_loss(buf_outputs, buf_logits)
            '''
            buf_inputs, buf_labels,buf_logits,buf_logits2,task_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            B,_,H,W =buf_inputs.shape

            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            targetsbuf_2 = torch.ones(B, 1, H, W).to(inputs.device)
            targetsbuf_2 = targetsbuf_2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            xbuf2 = torch.cat([buf_inputs, targetsbuf_2 / self.s], dim=1)
            inputs_buf = torch.cat([xbuf1, xbuf2], dim=0)
            labels_buf = torch.cat([buf_labels, buf_labels], dim=0)
            outputs_buf = self.net(inputs_buf)
            #logits_buf = torch.cat([buf_logits, buf_logits2], dim=0)

            loss += self.loss(outputs_buf, labels_buf)
            """
            if not self.first_task:
                B,_,H,W =buf_inputs.shape
                task_to_remove = self.task
                mask = task_labels != task_to_remove
                buf_inputs = buf_inputs[mask]
                buf_labels = buf_labels[mask]
                buf_logits = buf_logits[mask]
                task_labels=task_labels[mask]
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
            """
            #loss += 0.3 * F.mse_loss(outputs_buf, logits_buf)
           
            """
            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            buf_outputs1 = self.net(xbuf1)
            loss += self.loss(buf_outputs1, buf_labels)
            
           
            
            targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
            targets_buf2 = targets_buf2 * buf_labels.reshape(-1, 1, 1, 1) + 1
            x_buf2 = torch.cat([buf_inputs, targets_buf2 / self.s], dim=1)
            buf_outputs2=self.net(x_buf2)
            loss += self.loss(buf_outputs2, buf_labels)
            """
            

            
        
        '''
        if not self.buffer.is_empty() and self.net1 is not None and not self.first_task:
            buf_inputs,_, buf_logits,buf_task = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            B,_,H,W =buf_inputs.shape
            predictions = torch.nn.functional.softmax(buf_logits, dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
            targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            x_buf2 = torch.cat([buf_inputs, targets_buf2 / self.s], dim=1)
            buf_outputs=self.net(x_buf2)
            loss += 0.1 * F.mse_loss(buf_outputs, buf_logits)
            '''
        '''
            buf_inputs, buf_labels, _ ,_= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            buf_outputs = self.net(xbuf1)
            loss += 0.5 * self.loss(buf_outputs, buf_labels)
        '''
            

        loss.backward()
        self.opt.step()
        
        
        
        #self.buffer.add_data(examples=not_aug_inputs, labels=labels,logits=outputs_0.data)
        #self.buffer.add_data(examples=not_aug_inputs, labels=labels,logits=outputs.data,task_labels=(torch.ones(self.args.batch_size) *(self.task - 1)))
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:batch_size],logits=outputs[:batch_size].data,logits2= outputs[batch_size:].data,task_labels=(torch.ones(self.args.batch_size) *self.task))
        return loss.item()
   

    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)
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
        plt.savefig(f'tsne_visualization derid task:{self.task}.png', dpi=300, bbox_inches='tight')
        plt.close()
        """

        print('end_task call')

        
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
        else:
            self.old_model = self.deepcopy_model(self.net)
            #theta_0 = self.old_model.state_dict()
            #theta_1 = self.net.state_dict()
            #theta_interpolated = interpolate_weights(theta_0, theta_1, 0.4)
            #self.old_model.load_state_dict(theta_interpolated)
            #self.net.load_state_dict(theta_interpolated)

            


        
        
        #self.old_model = self.deepcopy_model(self.net)
        #self.nets[self.task]=self.deepcopy_model(self.net)
        
        torch.save(self.old_model, 'old_model.pt')
        torch.save(self.net, 'net.pt')



    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy

