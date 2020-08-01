import torch 
import torch.nn as nn 
from collections import OrderedDict
from torch.autograd import Variable

class ProtICU(nn.Module): 
    def __init__(self, input_shape, class_size, hidden_sizes, kernel_sizes, maxpool_size, one_by_one_sizes,
                 protop_classes, droprate, prototype_activation = 'log'):
        """
        input_shape(list): (batchsize, sequence length, feature dimensions)
        hidden_sizes(list): sizes of every hidden conv layer 
        kernel_sizes(list): sizes of kernel at every conv layer
        maxpool_size(int): sizes of maxpool (constant across different layers)
        one_by_one_sizes(list): sizes of every 1 x 1 Convolution layers 
        protop_classes(list): label of which prototype belongs to which class 
        """
        
        super(ProtICU, self).__init__()
        assert len(hidden_sizes) == len(kernel_sizes) 
        
        to_proto = []
        seqlen = input_shape[1]
        
        # Calculate length-preserving padding sizes 
        padding_sizes = []
        for k in kernel_sizes: 
            if k % 2 == 0: 
                padding_sizes.append((k // 2 - 1, k // 2))
            else: 
                padding_sizes.append(k // 2)
        
        # Conv Layers
        for i in range(len(hidden_sizes)): 
            if i == 0: 
                conv_layer = nn.Conv1d(input_shape[2], hidden_sizes[i], 
                                       kernel_sizes[i], stride=1,
                                       padding=padding_sizes[i])
                maxpool_layer = nn.MaxPool1d(maxpool_size,maxpool_size)
                to_proto.append(('conv'+str(i+1), conv_layer))
                to_proto.append(('relu_conv'+str(i+1), nn.ReLU()))
                to_proto.append(('maxpool'+str(i+1), maxpool_layer))
            else:
                conv_layer = nn.Conv1d(hidden_sizes[i-1], hidden_sizes[i], 
                                       kernel_sizes[i], stride=1,
                                       padding=padding_sizes[i])
                maxpool_layer = nn.MaxPool1d(maxpool_size,maxpool_size)
                to_proto.append(('conv'+str(i+1), conv_layer))
                to_proto.append(('relu_conv'+str(i+1), nn.ReLU()))
                to_proto.append(('maxpool'+str(i+1), maxpool_layer))
            seqlen = seqlen // maxpool_size
        
        # 1x1 Conv Layer 
        for i in range(len(one_by_one_sizes)): 
            if i == 0: 
                obo_layer = nn.Conv1d(hidden_sizes[-1], one_by_one_sizes[i], kernel_size=1)
                to_proto.append(('obo'+str(i+1), obo_layer))
                to_proto.append(('relu_obo'+str(i+1), nn.ReLU()))
            else: 
                obo_layer = nn.Conv1d(one_by_one_sizes[i-1], one_by_one_sizes[i], kernel_size=1)
                to_proto.append(('obo'+str(i+1), obo_layer))
                to_proto.append(('relu_obo'+str(i+1), nn.ReLU()))
        
        self.to_proto = nn.Sequential(OrderedDict(to_proto))
        
        # Prototypes 
        self.protop_classes = torch.Tensor(protop_classes) 
        self.prototypes = nn.Parameter(torch.rand((self.protop_classes.shape[0], one_by_one_sizes[-1])),
                                          requires_grad=False) # proto_num x dim
        self.prototype_activation = prototype_activation
        
        # Last Layer
        self.class_size = class_size
        self.last_layer = nn.Linear(self.protop_classes.shape[0], class_size, bias=False)
        
        # Initialise weights
        self.init_weights()
        
    def init_weights(self, incorrect_weight=-.5): 
        # initialise kaiming for Conv layers
        for m in self.to_proto.modules(): 
            if isinstance(m, nn.Conv1d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
        
        # initialise last layer
        _weights = torch.zeros((self.class_size, self.protop_classes.shape[0]), requires_grad=True)
        for i in range(self.class_size): 
            _weights[i,] =  (self.protop_classes == i).float() + \
                            (self.protop_classes != i).float() * incorrect_weight
        self.last_layer.weight.data.copy_(_weights)
      
    def l2_conv(self, protnet_out, protops): 
        """
        protnet_out(tensor): bs x patch_num x dim 
        protops(tensor): bs x protopsize x dim 
        """
        # I need to reassign to different names everytime to avoid in-place replacement
        diff = protnet_out.unsqueeze(2) - protops # bs x patch_num x protopsize x dim 
        l2_dis = diff * diff  
        l2_dis_sum = torch.sum(l2_dis, dim=3).transpose(1,2) # bs x protopsize x patch_num 
        l2_dis_rsum = torch.sqrt(l2_dis_sum)
        min_dis = torch.min(l2_dis_rsum, dim=2).values
        return min_dis 
    
    def dis2sim(self, dis, epsilon=.0001): 
        if self.prototype_activation == 'log': 
            sim = torch.log((dis + 1)/(dis + epsilon))
            return sim 
        elif self.prototype_activation == 'linear': 
            return -dis 
        else: 
            raise ValueError('prototype_activation not defined')

    def forward(self, input): 
        # Run models
        input_t = input.transpose(1,2) # transpose to bs x dim x sl 
        protnet_out = self.to_proto(input_t).transpose(1,2) # bs x patch_num x dim 
        min_dis = self.l2_conv(protnet_out, self.prototypes) # bs x proto_num 
        sim = self.dis2sim(min_dis)
        last = self.last_layer(sim)
        out = torch.nn.functional.softmax(last, dim=1)
        return out, min_dis
       
    def propose_prototype(self, data):
        X, y = Variable(data[0]), Variable(data[1]).reshape(-1)
        
        out = self.to_proto(X.transpose(1,2)).transpose(1,2) # bs x patch_num x dim 
        dis = out.unsqueeze(2) - self.prototypes # bs x patch_num x protopsize x dim 
        dis *= dis  
        dis = torch.sum(dis, dim = 3).transpose(1,2) # bs x protopsize x patch_num 
        dis = torch.sqrt(dis)

        # minimum across patches 
        min_dis = torch.min(dis, dim=2).values.data
        patch_loc = torch.argmin(dis, dim=2).data 

        # minimum across data points
        address_min = torch.zeros(self.protop_classes.shape).long()
        for c in range(self.protop_classes.unique().shape[0]):
            temp = min_dis.clone()
            temp[y != c] += 10000 # poison wrong class 
            address_min[self.protop_classes == c] = torch.argmin(temp, dim=0).data[self.protop_classes  == c] 
        proposal_min = min_dis[address_min,:].diag() # prot_num
        prot_raw = X[address_min,:,:] # prot_num x sl x dim 
        min_patch_loc = patch_loc[address_min,:].diag() # prot_num 
        prot_rep = out.data[address_min, min_patch_loc] # prot_num x dim 

        return proposal_min, prot_rep, prot_raw, min_patch_loc   
    
    def 
            
    def __name__(): 
        return 'ProtICU'
