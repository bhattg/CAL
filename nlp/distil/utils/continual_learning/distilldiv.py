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
import submarine 

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

def similarity(X, metric, divide_with=None, kernel_width=None):
    S = torch.cdist(X, X, p=2)
    if metric == 'rbf':
        # print(kernel_width)
        S=S**2
        if divide_with is None:
            return torch.exp(-S/kernel_width)
        else:
            return torch.exp(-S/(kernel_width*S.mean()))
    elif metric=='euclidean':
        return torch.max(S)-S
    else:
        print("Bad Metric!")
        sys.exit(1
)
def get_facility_location_submodular_order(X, mod_score, metric, B, c, smraiz_path="smraiz", smtk=1, no=0, stoch_greedy=0, lam=None, scores=None, divide_with=None, kernel_width=None, weights=None):
    '''
    Args
    - X: np.array, shape [N, M], design matrix
    - B: int, number of points to select
    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    '''
    # print('Computing facility location submodular order...')
    N = X.shape[0]
    # ct = time.time()
    X = similarity(X,metric, divide_with, kernel_width)
    # print(f"Similarity Construction Time {time.time()-ct}")
    # ct = time.time()
    X = X.cpu().numpy()
    # print("Transfer from GPU to CPU", time.time()-ct)

    # ct = time.time()
    fl = submarine.FacilityLocationFunctionExternalMatrix(X)
    scm = submarine.CalibratedDenseFeatureBasedFunction(submarine.VectorFloat(mod_score), 'LOG1P()')

    # mod = submarine.ModularFunction(submarine.VectorFloat(mod_score))
    # mix = submarine.MixtureSubmodularFunction(submarine.VectorDouble([lam, 1.0]),[mod, fl])
    mix = submarine.MixtureSubmodularFunction(submarine.VectorDouble([lam, 1.0]),[scm, fl])

    # print("FL instantiation time : ", time.time()-ct)

    sol_set = submarine.DefaultSet(N)
    sol_order = np.zeros(B).astype(np.int32)
    sol_gains = np.zeros(B).astype(np.float32)


    # StochasticAcceleratedGreedyConstrainedMaximization
    # ct = time.time()
    submarine.AcceleratedGreedyConstrainedMaximization(mix, B, sol_set, sol_order, sol_gains)    
    # print("Smraiz Python time", time.time()-ct)
    return sol_order, sol_set, sol_gains




class DistillDiv(Trainer):
    
    def __init__(self, training_dataset, val_set, net, args, continual_args, al_args=None, parser_args=None):


        super(DistillDiv, self).__init__(training_dataset=training_dataset, val_set=val_set, net=net, args=args, continual_args=continual_args, al_args=al_args, parser_args=parser_args)

        print(self.continual_args['target_classes'])
        self.logits = torch.zeros((len(self.current_task)), self.continual_args['target_classes'])
        self.al_args = al_args
        self.parser_args = parser_args
        self.metric = 'rbf'
        self.lam = self.continual_args['lambda']

        # Overwrite config file args with command line args if provided
        if parser_args:
            if parser_args.alpha:
                self.continual_args['alpha'] = np.float32(parser_args.alpha)
            if parser_args.beta:
                self.continual_args['beta'] = np.float32(parser_args.beta)

            if parser_args.kernel_width:
                self.continual_args['kernel_width'] = np.float32(parser_args.kernel_width)

            if parser_args.lam: 
                self.lam = np.float32(parser_args.lam)

        if self.metric == 'rbf':
            self.kernel_width = np.float32(self.continual_args['kernel_width'])


        self.buffer = Buffer(None, self.continual_args['C']) 

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
                if self.parser_args.bert:
                    x, y = (x[0].to(device=self.device), x[1].to(device=self.device)), y.to(device=self.device)
                else:
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


        new_coeff = 1



        # print(self.clf.device)
        for batch_id, (x, y, loader_idx) in enumerate(loader_tr):
            if not self.parser_args.bert:
                x, y = x.to(device=self.device), y.to(device=self.device)
            else:
                x, y = (x[0].to(device=self.device), x[1].to(device=self.device)), y.to(device=self.device)
            loss = 0

            ratio = self.buffer.get_size()/(len(loader_tr) * loader_tr.batch_size)
            old_coeff = 1 - 1/(ratio+1)
            new_coeff = 1/(ratio+1)
            if not self.buffer.is_empty():

                # Retrieve samples and dark knowledge
                past_x, past_y, past_z = self.buffer.retrieve_ER()
                if not self.parser_args:
                    past_x, past_y, past_z = past_x.to(self.device), past_y.to(self.device), past_z.to(self.device)
                else:
                    past_x, past_y, past_z = (past_x[0].to(self.device), past_x[1].to(self.device)), past_y.to(self.device), past_z.to(self.device)

                # print(past_x.shape)
                self.clf.eval()
                with torch.no_grad():
                    past_out = self.clf(past_x)
                    probs = F.softmax(past_out, dim=-1)
                    uncertainty_dist = 1 - torch.max(probs,dim=1)[0]

                    idx, _, _ = get_facility_location_submodular_order(past_out, uncertainty_dist, self.metric, self.continual_args['replay_size'], 0, kernel_width= (self.kernel_width), lam=self.lam)
                    # # uncertainty_dist = uncertainty_dist/torch.sum(uncertainty_dist)
                    # # idx = np.random.choice(np.array(range(100)), size=self.continual_args['replay_size'], p=uncertainty_dist, replace=False)
                    # idx = torch.multinomial(uncertainty_dist, self.continual_args['replay_size'])
                    # print(idx)
                    if self.parser_args.bert:
                        past_x = past_x[0][idx], past_x[1][idx]
                    else:
                        past_x = past_x[idx]
                    past_y = past_y[idx]
                    past_z = past_z[idx]


                self.clf.train()
                    # print(uncertainty.shape)
                # past_out = self.clf(past_x)
                # print(past_x.shape, past_y.shape, past_z.shape)

                # For real
                optimizer.zero_grad()
                past_out = self.clf(past_x)
                kl_loss = F.kl_div(F.softmax(past_out, dim=-1), past_z)
                ce_loss = criterion(past_out, past_y.long())
                loss += self.continual_args['alpha'] * kl_loss
                loss += self.continual_args['beta'] * ce_loss
                loss = loss * old_coeff

            optimizer.zero_grad()
            out = self.clf(x)
            if epoch == 1:
                self.logits[loader_idx] = F.softmax(out, dim=-1).detach().to(self.logits.device)


            loss += criterion(out, y.long()) * new_coeff
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            total += y.shape[0]
            loss.backward()

            optimizer.step()

        return accFinal / total, loss
