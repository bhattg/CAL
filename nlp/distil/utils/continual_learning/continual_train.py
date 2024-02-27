from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import sys
import wandb
import copy
import time

from distil.utils.models.cola_model import BERT_COLA

sys.path.append('../')  

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class AddIndexDataset(Dataset):
    
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset
        
    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data, label, index
    
    def __len__(self):
        return len(self.wrapped_dataset)


class Buffer:

    def __init__(self, dataset, num_samples=20):
        self.data = dataset
        self.num_samples = num_samples
        self.train_sampler = None 

        print(self.num_samples)
        if dataset:
            rand_sampler = torch.utils.data.RandomSampler(self.data, replacement=False)
            self.train_sampler = torch.utils.data.DataLoader(train, batch_size=self.num_samples, sampler=rand_sampler)


    def is_empty(self):
        if self.data:
            return False
        else:
            return True

    def get_size(self):
        if self.data:
            return len(self.data)
        else:
            return 0

    def update(self, dataset):

        self.data = dataset 

        if dataset:
            rand_sampler = torch.utils.data.RandomSampler(self.data, replacement=False)
            self.train_sampler = torch.utils.data.DataLoader(self.data, batch_size=self.num_samples, sampler=rand_sampler)

    def retrieve_ER(self):
        iterator = iter(self.train_sampler)
        ret_tuple = iterator.next()
        return ret_tuple




#custom training
class Trainer:

    """
    Provides a configurable training loop for AL.
    
    Parameters
    ----------
    training_dataset: torch.utils.data.Dataset
        The training dataset to use
    net: torch.nn.Module
        The model to train
    args: dict
        Additional arguments to control the training loop
        
        `batch_size` - The size of each training batch (int, optional)
        `islogs`- Whether to return training metadata (bool, optional)
        `optimizer`- The choice of optimizer. Must be one of 'sgd' or 'adam' (string, optional)
        `isverbose`- Whether to print more messages about the training (bool, optional)
        `isreset`- Whether to reset the model before training (bool, optional)
        `max_accuracy`- The training accuracy cutoff by which to stop training (float, optional)
        `min_diff_acc`- The minimum difference in accuracy to measure in the window of monitored accuracies. If all differences are less than the minimum, stop training (float, optional)
        `window_size`- The size of the window for monitoring accuracies. If all differences are less than 'min_diff_acc', then stop training (int, optional)
        `criterion`- The criterion to use for training (typing.Callable[], optional)
        `device`- The device to use for training (string, optional)
    """
    
    def __init__(self, training_dataset, val_set, net, args, continual_args, al_args=None, parser_args=None):

        # self.current_task should be an AddIndexDataset
        # self.buffer

        self.current_task = AddIndexDataset(training_dataset)
        self.val_set = val_set 
        self.net = net
        self.args = args
        self.continual_args = continual_args
        self.n_pool = len(training_dataset)
        self.wandb = False
        self.fixed_epoch = False 
        self.al_args = al_args
        self.parser_args = parser_args

        if parser_args:
            self.wandb = parser_args.wandb
            self.fixed_epoch = parser_args.fixed_epoch

            if parser_args.n_epoch:
                self.continual_args['n_epoch'] = np.float32(parser_args.n_epoch)
            if parser_args.lr:
                self.args['lr'] = np.float32(parser_args.lr)
            if parser_args.replay_size:
                self.continual_args['replay_size'] = int(parser_args.replay_size)


        if 'islogs' not in args:
            self.args['islogs'] = False

        if 'optimizer' not in args:
            self.args['optimizer'] = 'sgd'
        
        if 'isverbose' not in args:
            self.args['isverbose'] = False

        if 'max_accuracy' not in args:
            self.args['max_accuracy'] = 0.95

        if 'min_diff_acc' not in args: #Threshold to monitor for
            self.args['min_diff_acc'] = 0.01

        if 'window_size' not in args:  #Window for monitoring accuracies
            self.args['window_size'] = 10
            
        if 'criterion' not in args:
            self.args['criterion'] = nn.CrossEntropyLoss()
            
        if 'device' not in args:
            if parser_args:
                self.device = "cuda:" + parser_args.gpu 
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb


    def update_task(self, new_training_dataset):
        """
        Updates current task with newly labelled points

        """
        self.current_task = AddIndexDataset(new_training_dataset)


    def update_buffer(self):
        """
        Moves contents of current task to buffer, and clears current task
        
        """
        pass

    def get_acc_on_set(self, test_dataset):
        
        """
        Calculates and returns the accuracy on the given dataset to test
        
        Parameters
        ----------
        test_dataset: torch.utils.data.Dataset
            The dataset to test
        Returns
        -------
        accFinal: float
            The fraction of data points whose predictions by the current model match their targets
        """	
        
        try:
            self.clf
        except:
            self.clf = self.net

        if test_dataset is None:
            raise ValueError("Test data not present")
        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1 
        
        loader_te = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)
        self.clf.eval()
        accFinal = 0.

        with torch.no_grad():        
            self.clf = self.clf.to(device=self.device)
            for batch_id, (x,y) in enumerate(loader_te):     
                if self.parser_args.bert:
                    x, y = (x[0].to(device=self.device), x[1].to(device=self.device)), y.to(device=self.device)
                else:
                    x, y = x.to(device=self.device), y.to(device=self.device)
                out = self.clf(x)
                accFinal += torch.sum(1.0*(torch.max(out,1)[1] == y)).item() #.data.item()

        return accFinal / len(test_dataset)

    def evaluate_buffer(self):
        n = self.al_args['budget']

        # print(type(self.buffer.data.datasets[0]))
        accs = []
        for dataset in self.buffer.data.datasets:
            accs.append(self.get_acc_on_set(dataset))

        return accs


    def _train_weighted(self, epoch, loader_tr, optimizer, gradient_weights):
        pass


    def _train(self, epoch, loader_tr, optimizer):
        pass

    def check_saturation(self, acc_monitor):
        
        saturate = True

        for i in range(len(acc_monitor)):
            for j in range(i+1, len(acc_monitor)):
                if acc_monitor[j] - acc_monitor[i] >= self.args['min_diff_acc']:
                    saturate = False
                    break

        return saturate

    def train(self, gradient_weights=None, first_round=False, last_round=False):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


        def llf_reset(m):
            for name, layer in m.named_children(): 
                if name == 'linear':

                    for n, l in layer.named_modules():
                        print(n)
                        if hasattr(l, 'reset_parameters'):
                            print(f'Reset trainable parameters of layer = {l}')
                            l.reset_parameters()


        train_logs = []

        if first_round or last_round:
            n_epoch = self.continual_args['first_n_epoch']
        else:
            n_epoch = self.continual_args['n_epoch'] #self.args['n_epoch']
        
        if self.args['isreset'] and not first_round:
            self.clf = BERT_COLA().to(device=self.device) #self.net.apply(weight_reset).to(device=self.device)
        elif first_round:
            self.clf = BERT_COLA().to(device=self.device) #self.net.apply(weight_reset).to(device=self.device)
        else:
            # print("Trying")
            # print("No reset")
            
            if self.parser_args.partial_ws:
                print("Partial reset")
                net2 = copy.deepcopy(self.net)
                net2 = net2.apply(weight_reset).to(device=self.device)
                with torch.no_grad():
                    for real_parameter, random_parameter in zip(self.clf.parameters(), net2.parameters()):
                        real_parameter.mul_(.8).add_(random_parameter, alpha=.2)
            else:
                print("No reset")
                self.clf
            # self.clf = self.net.apply(llf_reset).to(device=self.device)
            


            # net2 = copy.deepcopy(self.net)
            # net2 = net2.apply(weight_reset).to(device=self.device)


            # # # SHRINK AND PERTURB
            # with torch.no_grad():
            #     for real_parameter, random_parameter in zip(self.clf.parameters(), net2.parameters()):
            #         real_parameter.mul_(.8).add_(random_parameter, alpha=.2)

            # for p in self.clf.parameters():
            #     print(p)
            #     break
            # except:
            #     print("Except")
            #     self.clf = self.net.apply(weight_reset).to(device=self.device)

        print(self.args['lr'])
        if self.args['optimizer'] == 'sgd':
            if first_round:
                optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
            elif last_round:
                optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)                
            else:
                # if self.args['cl_lr']:
                #     optimizer = optim.SGD(self.clf.parameters(), lr = self.args['cl_lr'] * (self.args['batch_size']), momentum=0.9, weight_decay=5e-4)
                #     lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
                # else:
                cl_lr = self.args['lr'] * ((self.args['batch_size'] +  self.continual_args['replay_size'])/(self.args['batch_size']))
                print(cl_lr)
                optimizer = optim.SGD(self.clf.parameters(), lr = cl_lr, momentum=0.9, weight_decay=5e-4)
                lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
                # lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95)        
        
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1


        # Set shuffle to true to encourage stochastic behavior for SGD
        if last_round:
            # z = torch.zeros((len(self.current_task)), self.continual_args['target_classes'])
            datasets = [AddIndexDataset(self.buffer.data.datasets[i].dataset) for i in range(len(self.buffer.data.datasets))] 
            old_data = ConcatDataset([self.current_task] +  datasets)
            loader_tr = DataLoader(old_data, batch_size=batch_size, shuffle=True, pin_memory=True)
            self.buffer = Buffer(None) # clear buffer
            print(len(loader_tr))
        else:
            loader_tr = DataLoader(self.current_task, batch_size=batch_size, shuffle=True, pin_memory=True)
            print(len(loader_tr))

        epoch = 1
        accCurrent = 0
        is_saturated = False
        acc_monitor = []
        # n_epoch = 100

        cur_best_net = copy.deepcopy(self.clf.state_dict())
        cur_best_acc = 0
        cur_best_ep = 0

        ct = time.time()
        while (epoch < n_epoch): # and accCurrent < self.args['max_accuracy']: #and (accCurrent < self.args['max_accuracy']) and (not is_saturated): 
            

            ct = time.time()
            # if last_round == True and accCurrent >= self.args['max_accuracy']:
            #     break
            if (first_round == True or last_round == True) and accCurrent >= .99: #self.args['max_accuracy']:
                print("Hello")
                break

            if not self.fixed_epoch and accCurrent >= self.args['max_accuracy']:
                break

            if gradient_weights is None:
                if last_round:
                    accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
                else:
                    accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            else:
                accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights)
            
            torch.cuda.synchronize()
            print("Per epoch time", time.time() - ct)
            if self.wandb:
                wandb.log({'Training Loss': lossCurrent})
                wandb.log({'Training Acc': accCurrent})

            # val_acc = self.get_acc_on_set(self.val_set)

            # if val_acc >= cur_best_acc:
            #     cur_best_net = copy.deepcopy(self.clf.state_dict())
            #     cur_best_acc = val_acc
            #     cur_best_ep = epoch



            # acc_monitor.append(val_acc)
            # print("Epoch:", str(epoch), "Val Acc", str(val_acc), "LR", str(optimizer.param_groups[0]['lr']))

            if self.args['optimizer'] == 'sgd':
                lr_sched.step()
            
            epoch += 1
            if(self.args['isverbose']):
                if epoch % 1 == 0:
                    print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)


            # if epoch - cur_best_ep >= 10:
            #     is_saturated = True

            # #Stop training if not converging
            # if len(acc_monitor) >= self.args['window_size']:

            #     is_saturated = self.check_saturation(acc_monitor)
            #     del acc_monitor[0]

            log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
            train_logs.append(log_string)
            


        # self.clf.load_state_dict(cur_best_net)

        print('Epoch:', str(epoch), 'Training accuracy:', round(accCurrent, 3), flush=True)

        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf
