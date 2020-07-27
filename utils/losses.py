import torch 
import numpy as np 

def cluster_loss(min_dis, y, protop_classes): 
    onehot = torch.nn.functional.one_hot(protop_classes.long(), protop_classes.unique().shape[0]).transpose(0,1)
    mask = torch.zeros(min_dis.shape)
    for i in range(onehot.shape[0]):
        mask[y.squeeze() == i,:] = onehot.float()[i]
    mask[mask == 0] = np.inf
    return torch.min(min_dis * mask, 1).values.mean()

def separation_loss(min_dis, y, protop_classes): 
    onehot = 1 - torch.nn.functional.one_hot(protop_classes.long(), protop_classes.unique().shape[0]).transpose(0,1)
    mask = torch.zeros(min_dis.shape)
    for i in range(onehot.shape[0]):
        mask[y.squeeze() == i,:] = onehot.float()[i]
    mask[mask == 0] = np.inf
    return -torch.min(min_dis * mask, 1).values.mean()