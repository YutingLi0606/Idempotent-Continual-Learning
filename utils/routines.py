import os
import torch
import torch.nn as nn
import torch.utils.data as Data

from backbone import MammothBackbone


def forward_loader_all_layers(model: MammothBackbone, loader: Data.DataLoader, device='cuda'):
    '''
    Forward the entire loader, return logits and features of all layers
    Always use the non-augmented images
    '''
    model = model.to(device)
    model.eval()

    logits = [] 
    feats = [[] for _ in range(len(model.net_channels))]
    xs = []
    ys = []

    for batch in loader:
        if len(batch) == 3:
            x, y, x_notaug = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        elif len(batch) == 2:
            x, y = batch[0].to(device), batch[1].to(device)
            x_notaug = x

        with torch.no_grad():
            x_logits, x_feats = model.forward_all_layers(x_notaug)
        
        logits.append(x_logits.detach().cpu())
        for i in range(len(feats)):
            feats[i].append(x_feats[i].detach().cpu())
        xs.append(x_notaug.cpu())
        ys.append(y.cpu())

    logits = torch.cat(logits, dim=0) # (n, n_classes)
    for i in range(len(feats)):
        feats[i] = torch.cat(feats[i], dim=0)
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    return logits, feats, xs, ys

def forward_loader(model, loader, device='cuda'):
    '''
    Forward the entire loader, return logits, features, images and labels.
    Always use the non-augmented images
    '''
    model = model.to(device)
    
    logits = []
    feats = []
    xs = []
    ys = []
    for batch in loader:
        x, y, x_notaug = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        with torch.no_grad():
            x_logits, x_feats = model(x_notaug, returnt='all')
        
        logits.append(x_logits.detach().cpu())
        feats.append(x_feats.detach().cpu()) 
        xs.append(x_notaug.cpu())
        ys.append(y.cpu())

    logits = torch.cat(logits, dim=0) # (n, n_classes)
    feats = torch.cat(feats, dim=0) # (n, n_feats)
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    return logits, feats, xs, ys