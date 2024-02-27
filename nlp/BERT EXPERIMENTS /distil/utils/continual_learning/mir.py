from .continual_train import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import sys
import copy


class MIR(Trainer):
    
    def __init__(self, training_dataset, val_set, net, args, continual_args, al_args, parser_args=None):


        super(MIR, self).__init__(training_dataset=training_dataset, val_set=val_set, net=net, args=args, continual_args=continual_args, al_args=al_args, parser_args=parser_args)

        self.buffer = Buffer(None, self.continual_args['C']) 
        


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
            if self.parser_args.bert:
                x, y = (x[0].to(device=self.device), x[1].to(device=self.device)), y.to(device=self.device)
            else:
                x, y = x.to(device=self.device), y.to(device=self.device)
            loss = 0
            if not self.buffer.is_empty():

                # Subsample from buffer
                past_x, past_y, _ = self.buffer.retrieve_ER()
                if self.parser_args.bert:
                    past_x, past_y = (past_x[0].to(device=self.device), past_x[1].to(device=self.device)), past_y.to(device=self.device)
                else:
                    past_x, past_y = past_x.to(device=self.device), past_y.to(device=self.device)

                # Compute initial loss
                with torch.no_grad():
                    out1 = self.clf(past_x)
                    old_loss = F.cross_entropy(out1, past_y, reduce=False)

                # Save old model and optimizer
                optimizer.zero_grad()
                old_opt = copy.deepcopy(optimizer.state_dict())
                old_net = copy.deepcopy(self.clf.state_dict())

                # Take virtual step
                virtual_loss = criterion(self.clf(x), y.long())
                virtual_loss.backward()
                optimizer.step(); torch.cuda.empty_cache()

                # Compute new loss
                with torch.no_grad():
                    out2 = self.clf(past_x)
                    new_loss = F.cross_entropy(out2, past_y, reduce=False)

                mir_score =  F.softmax((new_loss - old_loss), dim=0).squeeze() #new_loss - old_loss 
                mir_score = mir_score + 1e-8; mir_score = mir_score/mir_score.sum()
                idx = np.random.choice(len(mir_score), self.continual_args['replay_size'], p = mir_score.cpu().numpy(), replace=False)

                # Restore old model and optimizer
                optimizer.load_state_dict(old_opt)
                self.clf.load_state_dict(old_net)

                # Create batch
                if self.parser_args.bert:
                    x = (torch.cat([x[0], past_x[0][idx].to(device=self.device)]),  torch.cat([x[1], past_x[1][idx].to(device=self.device)]))
                else:
                    x = torch.cat([x, past_x[idx].to(device=self.device)])
                y = torch.cat([y, past_y[idx].to(device=self.device)])

            optimizer.zero_grad()
            out = self.clf(x);# print(x[0].shape)

            loss += criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            total += y.shape[0]
            loss.backward()

            optimizer.step()

        return accFinal / total, loss
