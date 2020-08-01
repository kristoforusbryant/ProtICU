import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import sklearn.metrics as metrics

class TrainTest(): 
    def __init__(self, model, data, params, push_epochs):
        # hyperparams
        self.batch_size      = int(params['BATCH_SIZE'])
        self.epochs          = params['EPOCHS']
        self.loss            = params['LOSS']
        self.early_stopping  = params['EARLY_STOPPING']
        self.patience        = params['PATIENCE']
        self.min_delta       = params['MIN_DELTA']
        self.use_gpu         = torch.cuda.is_available()
        
        # data 
        dataset = TensorDataset(data[0][0], data[0][1])
        self.trainset = DataLoader(dataset, self.batch_size,
                                shuffle=True, drop_last=True)
        dataset = TensorDataset(data[1][0], data[1][1])
        self.valset = DataLoader(dataset, self.batch_size,
                                shuffle=True, drop_last=True)
        self.testset = data[2]
        del dataset
        
        # model 
        if model.__name__ == 'CNN_1D':
            self.model = model(input_shape  = data[0][0].shape, 
                               class_size   = int(data[2][1].unique().shape[0]), 
                               hidden_sizes = params['HIDDEN_SIZES'], 
                               kernel_sizes = params['KERNEL_SIZES'], 
                               maxpool_size = params['MAXPOOL'],
                               fc_sizes     = params['FC_SIZES'],
                               droprate     = params['DROPOUT']) 
        elif model.__name__ == 'ProtICU':
            # generating protop_classes
            protop_classes = np.zeros(params['PROTOTYPE_NUM'])
            class_size = int(data[2][1].unique().shape[0])
            largest_mult = int(protop_classes.shape[0] // class_size)
            for i in range(class_size-1): 
                protop_classes[i * largest_mult: (i + 1) * largest_mult] = i 
            protop_classes[(class_size-1) * largest_mult: ] = class_size-1

            # initialising model 
            self.model = model(input_shape      = data[0][0].shape,
                               class_size       = class_size,
                               hidden_sizes     = params['HIDDEN_SIZES'],
                               kernel_sizes     = params['KERNEL_SIZES'],
                               maxpool_size     = params['MAXPOOL'],
                               one_by_one_sizes = params['OBO_SIZES'],
                               protop_classes   = protop_classes,
                               droprate         = params['DROPOUT'], 
                               prototype_activation = params['PROTO_ACTIVATION'])
            
        self.optimizer = params['OPTIMIZER'](self.model.parameters(),
                                            lr=params['LEARNING_RATE'])
        
        
        #prototype related 
        self.prototype_raw   = (None, None)
        self.prototype_num   = params['PROTOTYPE_NUM']
        self.push_epochs     = push_epochs
        
        # result
        self.mean_train_loss = []
        self.mean_val_loss   = []
        self.min_val_loss    = np.infty
        self.stats           = {}
        
        
    
    def train_one(self, data):
        if self.use_gpu:
            X, y = Variable(data[0].cuda()), Variable(data[1].cuda())
        else: 
            X, y = Variable(data[0]), Variable(data[1])
        
        self.model.zero_grad()
        
        if 'protop_loss' in self.loss.__repr__(): #self.loss.__name__() == 'Prototype Loss':
            outputs, min_dis = self.model(X)
            loss = self.loss(outputs, y.reshape(-1).long(), min_dis, self.model.protop_classes) 
        else:
            outputs = self.model(X)
            loss = self.loss(outputs, y.reshape(-1).long())
        
        if self.model.training:
            #print('Conv_1D_1') 
            #print(next(self.model.to_proto.modules())[3].weight)
            
            #print('Last_linear')
            #print(self.model.last_layer.weight)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.data
    
    def train(self): 
        wait = 0
        #print(self.model.prototypes)
        for epoch in np.arange(self.epochs): 
            # training
            train_loss = []
            self.model.train()
            for data in tqdm(self.trainset): train_loss.append(self.train_one(data))
            self.mean_train_loss.append(np.mean(train_loss))
            
            # validating
            val_loss = []
            self.model.eval()
            for data in tqdm(self.valset): val_loss.append(self.train_one(data))
            
            # pushing 
            if epoch in self.push_epochs:
                print('Updating prototype representation')
                closest = [[np.inf, None, None, None] for _ in range(self.prototype_num)] 
                for data in tqdm(self.trainset):
                    print([closest[i][0] for i in range(len(closest))])
                    # proposal_min is proto_num x 1
                    proposal_min, prototype_rep, prototype_raw, patch_loc = self.model.propose_prototype(data) 
                    for i in range(self.prototype_num): 
                        if proposal_min[i] < closest[i][0]: 
                            closest[i][0] = proposal_min[i]
                            closest[i][1] = prototype_rep[i]
                            closest[i][2] = prototype_raw[i]
                            closest[i][3] = patch_loc[i]
                            
                #print(self.model.prototypes.shape)
                #print(self.model.prototypes)
                for i in range(self.prototype_num):
                    self.model.prototypes[i,:] = closest[i][1].clone() + 1e-6 # update prototype representation
                #print(self.model.prototypes.shape)
                #print(self.model.prototypes)
                
                self.prototype_raw = ([closest[i][2] for i in range(len(closest))]
                                      ,[closest[i][3] for i in range(len(closest))]) # update raw prototype
                
            # check whether to early_stop
            if self.early_stopping and self.min_val_loss < np.mean(val_loss):
                if self.patience <= wait:
                    print('Early stopped at Epoch: ', epoch)
                    self.stats['epoch_stopped'] = epoch 
                    break 
                else:
                    wait += 1
                    print('val loss increased, patience count: ', wait)
            else:
                wait = 0
                self.min_val_loss = min(self.min_val_loss, np.mean(val_loss))
                    
            self.mean_val_loss.append(np.mean(val_loss))
            
            # printing 
            #print(self.model.prototypes)
            print([(self.model.prototypes[i] > 1e-7).sum() for i in range(self.model.prototypes.shape[0])])
            print('Epoch: {}, train_loss: {}, valid_loss: {}'.format( \
                    epoch, \
                    np.around(np.mean(train_loss), decimals=8),\
                    np.around(np.mean(val_loss), decimals=8)))
            print(self.model.last_layer.weight)
            
        return 0 
    
    def test(self): 
        if self.use_gpu: 
            X, y = Variable(self.testset[0].cuda()), Variable(self.testset[1].cuda())
        else: 
            X, y = Variable(self.testset[0]), Variable(self.testset[1])
        
        if type(self.model).__name__ == 'ProtICU':
            outputs  = self.model(X)[0].data
        else: 
            outputs = self.model(X).data
            
        self.stats['auroc'] = metrics.roc_auc_score(y, outputs[:,1])
        self.stats['auprc'] = metrics.average_precision_score(y, outputs[:,1])
        self.stats['acc']   = metrics.accuracy_score(y, np.around(outputs[:,1]))
        self.stats['f1']    = metrics.f1_score(y, np.around(outputs[:,1]))
        
        return self.stats 