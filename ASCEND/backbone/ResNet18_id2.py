# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu

from backbone import MammothBackbone

class Classifier(torch.nn.Module):
    def __init__(self,
                 feat_dim,
                 nb_cls,
                 cos_temp):
        super(Classifier, self).__init__()

        fc = torch.nn.Linear(feat_dim, nb_cls)        
        self.weight = torch.nn.Parameter(fc.weight.t(), requires_grad=True)
        self.bias = torch.nn.Parameter(fc.bias, requires_grad=True)
        self.cos_temp = torch.nn.Parameter(torch.FloatTensor(1).fill_(cos_temp), requires_grad=False)
        self.apply = self.apply_cosine
    def get_weight(self):
        return self.weight, self.bias

    def apply_cosine(self, feature, weight, bias):
        
        feature = F.normalize(feature, p=2, dim=1, eps=1e-12) ## Attention: normalized along 2nd dimension!!!
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)## Attention: normalized along 1st dimension!!!

        cls_score = self.cos_temp * (torch.mm(feature, weight))
        return cls_score


    def forward(self, feature):
        weight, bias = self.get_weight()
        cls_score = self.apply(feature, weight, bias)

        return cls_score
def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet1(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet1, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.final_d = nf * 8
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
     

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """

        out = relu(self.bn1(self.conv1(x))) # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16

        return out

 

class ResNet2(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int,use_cos=False) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet2, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.final_d = nf * 8
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.net_channels = [nf * 1, nf * 2, nf * 4, nf * 8]
        self.y_hat_fc = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.LeakyReLU()
        )
        if  use_cos:
            self.classifier = Classifier(512*block.expansion, num_classes, 12)
            print("use cos!")
        else:
            self.classifier = self.linear
    

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, returnt2='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """

       
        out = x + self.y_hat_fc(y)[..., None, None]
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        feat = out
        out = avg_pool2d(out, out.shape[2]) # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512

        out = self.classifier(feature)
        if returnt2=="tsne":
            return feature
        else:
            return out[:, :self.num_classes], feat
        



 


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int,use_cos=False) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.f1 = ResNet1(BasicBlock, [2, 2, 2, 2], num_classes, nf)
        self.f2 = ResNet2(BasicBlock, [2, 2, 2, 2], num_classes, nf,use_cos)
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.final_d = nf * 8
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, returnt='features') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """
        z = self.f1(x)
        
        if returnt=='out':
            y_pred, z_pred  = self.f2(z, y,returnt2=returnt)
            return y_pred, z_pred
        if  returnt == 'tsne':
            feature  = self.f2(z, y,returnt2=returnt)
            return feature
        

def resnet18_id2(nclasses: int, nf: int=64,use_cos=False) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """

    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf,use_cos)
