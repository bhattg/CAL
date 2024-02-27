from .continual_train import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

class DERDataset(Dataset):

    def __init__(self, dataset, z):
        self.dataset = dataset
        self.z = z 

    def __getitem__(self, index):
        data, label = self.dataset[index]
        logit = self.z[index]
        return data, label, logit

    def __len__(self):
        return len(self.dataset)

class DER(Trainer):
    
    def __init__(self, training_dataset, val_set, net, args, continual_args, al_args=None, parser_args=None):

        # super(Distill, self).__init__(training_dataset=training_dataset, val_set=val_set, net=net, args=args, continual_args=continual_args, al_args=al_args, parser_args=parser_args)

        super(DER, self).__init__(training_dataset=training_dataset, val_set=val_set, net=net, args=args, continual_args=continual_args, al_args=al_args, parser_args=parser_args)


        self.logits = torch.zeros((len(self.current_task)), self.continual_args['target_classes'])
        self.al_args = al_args


        # Overwrite config file args with command line args if provided
        if parser_args:
            if parser_args.alpha:
                self.continual_args['alpha'] = np.float32(parser_args.alpha)
            if parser_args.beta:
                self.continual_args['beta'] = np.float32(parser_args.beta)

        self.buffer = Buffer(None, self.continual_args['replay_size']) 

    def update_buffer(self):
        """
        Moves contents of current task to buffer, and clears current task
        
        """
        if self.buffer.is_empty():
            new_buffer_data = DERDataset(self.current_task.wrapped_dataset, self.logits)
            self.buffer.update(ConcatDataset([new_buffer_data]))
        else:
            new_buffer_data = DERDataset(self.current_task.wrapped_dataset, self.logits)
            self.buffer.update(ConcatDataset([new_buffer_data] + self.buffer.data.datasets))

        self.current_task = None


        # print(self.buffer.data.labels)
        
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
            for pack in loader_te: 
                x,y = pack[0],pack[1]   
                x, y = x.to(device=self.device), y.to(device=self.device)
                out = self.clf(x)
                accFinal += torch.sum(1.0*(torch.max(out,1)[1] == y)).item() #.data.item()

        return accFinal / len(test_dataset)


    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        accFinal = 0.
        total = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"

        # print(self.clf.device)
        for batch_id, (x, y, loader_idx) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)
            loss = 0

            if not self.buffer.is_empty():

                # Retrieve samples and dark knowledge
                past_x, past_y, past_z = self.buffer.retrieve_ER()
                past_x, past_y, past_z = past_x.to(self.device), past_y.to(self.device), past_z.to(self.device)
                # past_out = self.clf(past_x)


                # For grad logging

                # # MSE
                # optimizer.zero_grad(set_to_none=True)
                # past_out = self.clf(past_x)
                # mse_loss = F.mse_loss(past_out, past_z) #criterion(past_out, torch.unsqueeze(past_y[i].long(),dim=0))
                # mse_loss.backward(retain_graph=True)
                # mse_grad = self.continual_args['alpha'] * torch.unsqueeze(self.clf.get_grads(),dim=0).cpu().numpy()
                # mse_grad_norm = np.linalg.norm(mse_grad)

                # # CE
                # optimizer.zero_grad(set_to_none=True)
                # past_out = self.clf(past_x)
                # ce_loss = criterion(past_out, past_y.long()) #criterion(past_out, torch.unsqueeze(past_y[i].long(),dim=0))
                # ce_loss.backward(retain_graph=True)
                # ce_grad = self.continual_args['alpha'] * torch.unsqueeze(self.clf.get_grads(),dim=0).cpu().numpy()
                # ce_grad_norm = np.linalg.norm(ce_grad)

                # # Log
                # wandb.log({'MSE Grad Norm': mse_grad_norm})
                # wandb.log({'CE Grad norm': ce_grad_norm})
                # wandb.log({'MSE Loss': mse_loss.item()})
                # wandb.log({'CE Loss': ce_loss.item()})
                # wandb.log({'Scaled MSE Loss': self.continual_args['alpha'] * mse_loss.item()})
                # wandb.log({'Scaled CE Loss': self.continual_args['beta'] * ce_loss.item()})

                # For real
                optimizer.zero_grad()
                past_out = self.clf(past_x)
                mse_loss = F.mse_loss(past_out, past_z)
                ce_loss = criterion(past_out, past_y.long())
                loss += self.continual_args['alpha'] * mse_loss
                loss += self.continual_args['beta'] * ce_loss

            optimizer.zero_grad()
            out = self.clf(x)


            if epoch == 1:

                self.logits[loader_idx] = out.detach().to(self.logits.device)

            loss += criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            total += x.shape[0]
            loss.backward()

            optimizer.step()

        return accFinal / total, loss
