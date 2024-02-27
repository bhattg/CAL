from .continual_train import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import sys
import copy


class ER(Trainer):
    
    def __init__(self, training_dataset, val_set, net, args, continual_args, al_args, parser_args=None):


        super(ER, self).__init__(training_dataset=training_dataset, val_set=val_set, net=net, args=args, continual_args=continual_args, al_args=al_args, parser_args=parser_args)
        

        self.buffer = Buffer(None, self.continual_args['replay_size']) 



    def update_buffer(self):
        """
        Moves contents of current task to buffer, and clears current task
        
        """
        if self.buffer.is_empty():
            self.buffer.update(self.current_task)
        else:
            self.buffer.update(ConcatDataset([self.current_task, self.buffer.data]))

        self.current_task = None


    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        accFinal = 0.
        total = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"

        for batch_id, (x, y, loader_idx) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)
            loss = 0
            
            if not self.buffer.is_empty():
                # Retrieve samples
                past_x, past_y, _ = self.buffer.retrieve_ER()
                x = torch.cat([x, past_x.to(device=self.device)])
                y = torch.cat([y, past_y.to(device=self.device)])

            optimizer.zero_grad()
            out = self.clf(x)

            loss += criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            total += x.shape[0]
            loss.backward()

            optimizer.step()

        return accFinal / total, loss
