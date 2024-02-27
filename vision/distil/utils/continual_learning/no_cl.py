from .continual_train import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import sys
import copy


class No_CL(Trainer):
    
    def __init__(self, training_dataset, val_set, net, args, continual_args, al_args=None, parser_args=None):


        super(No_CL, self).__init__(training_dataset, val_set, net, args, continual_args, al_args, parser_args)
        self.buffer = Buffer(None, self.continual_args['replay_size']) 
        
        # Overwrite config file args with command line args if provided
        if parser_args:
            if parser_args.lr:
                self.args['lr'] = np.float32(parser_args.lr)
            if parser_args.warm_start:
                self.args['isreset'] = False
            else:
                self.args['isreset'] = True

    def update_buffer(self):
        """
        Moves contents of current task to buffer, and clears current task
        
        """
        pass

    def update_task(self, new_training_dataset):
        """
        Updates current task with newly labelled points

        """
        self.current_task = ConcatDataset([AddIndexDataset(new_training_dataset), self.current_task]) 


    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        accFinal = 0.
        total = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"

        for batch_id, (x, y, loader_idx) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)
            loss = 0
            
            optimizer.zero_grad()
            out = self.clf(x)

            loss += criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            total += x.shape[0]
            loss.backward()

            optimizer.step()

        return accFinal / total, loss
