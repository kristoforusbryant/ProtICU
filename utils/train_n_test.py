import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import sklearn.metrics as metrics

class TrainTest(): 
    def __init__(self, model, data, params):
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
        if model.__str__() == '1D-CNN': 
            self.model = model(input_shape  = data[0][0].shape, 
                               class_size   = data[2][1].shape[1], 
                               hidden_sizes = params['HIDDEN_SIZES'], 
                               kernel_sizes = params['KERNEL_SIZES'], 
                               maxpool_size = params['MAXPOOL'],
                               fc_sizes     = params['FC_SIZES'],
                               droprate     = params['DROPOUT']) 
        elif model.__str__() == 'ProtICU': 
            # generating protop_classes
            protop_classes = np.array([])
            class_size = data[2][1].shape[1]
            for i in range(class_size):
                protop_classes = np.array(protop_classes, [i] * (params['OBO_SIZES'][-1] // class_size))
            protop_classes[params['OBO_SIZES'][-1]:] = class_size - 1 # protop_classes
            
            # initialising model 
            self.model = model(input_shape      = data[0][0].shape, 
                               class_size       = class_size, 
                               hidden_sizes     = params['HIDDEN_SIZES'], 
                               kernel_sizes     = params['KERNEL_SIZES'], 
                               maxpool_size     = params['MAXPOOL'],
                               one_by_one_sizes = params['OBO_SIZES'],
                               protop_classes   = protop_classes, 
                               droprate         = params['DROPOUT'])             
            
        self.optimizer = params['OPTIMIZER'](self.model.parameters(),
                                            lr=params['LEARNING_RATE'])
        
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
        
        if loss.__str__() == 'Prototype Loss': 
            outputs, min_dis = self.model(X)
            loss = self.loss(outputs, y.reshape(-1).long(), min_dis, self.protop_classes) 
        else: 
            outputs = self.model(X)
            loss = self.loss(outputs, y.reshape(-1).long())
        
        if self.model.training: 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.data
    
    def train(self): 
        wait = 0
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
            print('Epoch: {}, train_loss: {}, valid_loss: {}'.format( \
                    epoch, \
                    np.around(np.mean(train_loss), decimals=8),\
                    np.around(np.mean(val_loss), decimals=8)))
        return 0 
    
    def test(self): 
        if self.use_gpu: 
            X, y = Variable(self.testset[0].cuda()), Variable(self.testset[1].cuda())
        else: 
            X, y = Variable(self.testset[0]), Variable(self.testset[1])
        
        outputs  = self.model(X).data
        self.stats['auroc'] = metrics.roc_auc_score(y, outputs[:,1])
        self.stats['auprc'] = metrics.average_precision_score(y, outputs[:,1])
        self.stats['acc']   = metrics.accuracy_score(y, np.around(outputs[:,1]))
        self.stats['f1']    = metrics.f1_score(y, np.around(outputs[:,1]))
        
        return self.stats 