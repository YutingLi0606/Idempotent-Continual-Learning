import torch
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import copy
import torch.nn.functional as F
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACEid(ContinualModel):
    NAME = 'er_aceid'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACEid, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0
        self.c=0
        self.s=backbone.num_classes
        self.first_task=True
        self.net1=None
        self.ft=True
        self.task=0

 

    def observe(self, inputs, labels, not_aug_inputs):
        batch_size, _, H, W = inputs.shape
        targets_1 = self.c * torch.ones(batch_size, 1, H, W).to(inputs.device)
        x1 = torch.cat([inputs, targets_1 / self.s], dim=1)
        targets_2 = torch.ones(batch_size, 1, H, W).to(inputs.device)
        targets_2 = targets_2 * labels.reshape(-1, 1, 1, 1) + 1
        x2 = torch.cat([inputs, targets_2 / self.s], dim=1)
        inputs_sum = torch.cat([x1, x2], dim=0)
        labels_sum = torch.cat([labels, labels], dim=0)

        
        #loss = self.loss(outputs, labels_sum)

        
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        #logits = self.net(inputs)
        logits = self.net(inputs_sum)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels_sum)
        loss_re = torch.tensor(0.)  

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
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

            loss_re = self.loss(outputs_buf, labels_buf)
            
            #targetsbuf_1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            #xbuf1 = torch.cat([buf_inputs, targetsbuf_1 / self.s], dim=1)
            #outputs_buf = self.net(xbuf1)

            #loss_re = self.loss(outputs_buf, buf_labels)
            
            
            targets_buf1 = self.c * torch.ones(B, 1, H, W).to(inputs.device)
            x_buf1 = torch.cat([buf_inputs, targets_buf1 / self.s], dim=1)
            buf_logits1 = self.net(x_buf1)
            predictions = torch.nn.functional.softmax(buf_logits1, dim=1)
            pred_labels = predictions.argmax(dim=1).to(inputs.device)
            targets_buf2 = torch.ones(B, 1, H, W).to(inputs.device)
            targets_buf2 = targets_buf2 * pred_labels.reshape(-1, 1, 1, 1) + 1
            x_buf2 = torch.cat([buf_inputs, targets_buf2 / self.s], dim=1)
            buf_outputs=self.net(x_buf2)
            
            
            loss += 0.3*F.mse_loss(buf_outputs, buf_logits1)
            
            
        

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()
    def end_task(self, dataset):
        print('\n\n')
        self.task+=1
        print(self.task)

        print('end_task call')

        
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
            print("not first now")
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
