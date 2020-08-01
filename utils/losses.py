import torch 
import numpy as np 

class protop_loss(): 
    def __init__(self, class_weights, clust_reg, sep_reg):
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.clust_reg = clust_reg
        self.sep_reg = sep_reg 
    
    def cluster_loss(self, min_dis, y, protop_classes): 
        onehot = torch.nn.functional.one_hot(protop_classes.long(), protop_classes.unique().shape[0]).transpose(0,1)
        mask = torch.zeros(min_dis.shape)
        for i in range(onehot.shape[0]):
            mask[y.squeeze() == i,:] = onehot.float()[i]
        mask[mask == 0] += 10 # find a better way to mask!
        loss,_ = torch.min(min_dis * mask, dim=1)
        return torch.mean(loss)

    def separation_loss(self, min_dis, y, protop_classes): 
        onehot = 1 - torch.nn.functional.one_hot(protop_classes.long(), protop_classes.unique().shape[0]).transpose(0,1)
        mask = torch.zeros(min_dis.shape)
        for i in range(onehot.shape[0]):
            mask[y.squeeze() == i,:] = onehot.float()[i]
        mask[mask == 0] += 10
        loss,_ = torch.min(min_dis * mask, dim=1) 
        return -torch.mean(loss)
    
    def __call__(self, out, y, min_dis, protop_classes, verbose=False): 
        ce = self.ce(out,y)
        clust = self.cluster_loss(min_dis, y, protop_classes) * self.clust_reg
        sep = self.separation_loss(min_dis, y , protop_classes) * self.sep_reg
        loss =  ce + clust + sep 
        if verbose: 
            print(ce)
            print(clust)
            print(sep)
        return loss
    
    def __repr__(self): 
        return 'protop_loss_' + str(self.clust_reg) +'_'+ str(self.sep_reg)
    
    def __name__(self): 
        return 'Prototype Loss'
