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
        mask[mask == 0] = np.inf
        return torch.min(min_dis * mask, 1).values.mean()

    def separation_loss(self, min_dis, y, protop_classes): 
        onehot = 1 - torch.nn.functional.one_hot(protop_classes.long(), protop_classes.unique().shape[0]).transpose(0,1)
        mask = torch.zeros(min_dis.shape)
        for i in range(onehot.shape[0]):
            mask[y.squeeze() == i,:] = onehot.float()[i]
        mask[mask == 0] = np.inf
        return -torch.min(min_dis * mask, 1).values.mean()
    
    def __call__(self, out, min_dis, y, protop_classes, verbose=False): 
        loss =  self.ce(out, y.reshape(-1).long())\
                + self.cluster_loss(min_dis, y, protop_classes) \
                + self.separation_loss(min_dis, y , protop_classes)
        print(self.ce(out, y.reshape(-1).long()))
        print(self.cluster_loss(min_dis, y, protop_classes))
        print(self.separation_loss(min_dis, y , protop_classes))
        return loss
    
    def __str__(): 
        return 'Prototype Loss'