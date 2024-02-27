from .continual_train import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import sys
import copy


class No_Replay(Trainer):
    
    def __init__(self, training_dataset, val_set, net, args, continual_args, al_args=None, parser_args=None):


        super(No_Replay, self).__init__(training_dataset, val_set, net, args, continual_args, al_args, parser_args)
        
        # # Overwrite config file args with command line args if provided
        # if parser_args:
        #     self.args['lr'] = np.float32(parser_args.lr)

        self.buffer = Buffer(None)
        print(args)

    def update_buffer(self):

        """
        Moves contents of current task to buffer, and clears current task
        
        """
# self.buffer.update(ConcatDataset([new_buffer_data]))

        if self.buffer.is_empty():
            self.buffer.update(ConcatDataset([self.current_task.wrapped_dataset]))
        else:
            self.buffer.update(ConcatDataset([self.current_task.wrapped_dataset] + self.buffer.data.datasets))

        self.current_task = None


    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        accFinal = 0.
        total = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"

        for batch_id, (x, y, loader_idx) in enumerate(loader_tr):
            # torch.save(x, "X_noreplay.pth"); sys.exit()
            x, y = (x[0].to(device=self.device),x[1].to(device=self.device)), y.to(device=self.device)
            loss = 0
            
            optimizer.zero_grad()
            out = self.clf(x)

            loss += criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            total += y.shape[0]
            loss.backward()

            optimizer.step()

        return accFinal / total, loss
