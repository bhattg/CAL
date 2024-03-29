from .entropy_sampling import EntropySampling
from .least_confidence_sampling import LeastConfidenceSampling
from .margin_sampling import MarginSampling
# from .submod_sampling import SubmodularSampling
from .strategy import Strategy
import time 
import numpy as np
import os
from torch.utils.data import Subset
import torch
import submarine 

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
def get_facility_location_submodular_order(X, mod_score, metric, B, c, smraiz_path="./", smtk=1, no=0, stoch_greedy=0, lam=None, scores=None, divide_with=None, kernel_width=None, weights=None):
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
    X = similarity(X.cpu(),metric, divide_with, kernel_width)
    # print(f"Similarity Construction Time {time.time()-ct}")
    # ct = time.time()
    X = X.cpu().numpy()
    # print("Transfer from GPU to CPU", time.time()-ct)

    # ct = time.time()
    fl = submarine.FacilityLocationFunctionExternalMatrix(X)
    mod = submarine.ModularFunction(submarine.VectorFloat(mod_score))

    mix = submarine.MixtureSubmodularFunction(submarine.VectorDouble([lam, 1.0]),[mod, fl])

    # print("FL instantiation time : ", time.time()-ct)

    sol_set = submarine.DefaultSet(N)
    sol_order = np.zeros(B).astype(np.int32)
    sol_gains = np.zeros(B).astype(np.float32)


    # StochasticAcceleratedGreedyConstrainedMaximization
    # ct = time.time()
    submarine.AcceleratedGreedyConstrainedMaximization(mix, B, sol_set, sol_order, sol_gains) 
    print(sol_gains)   
    # print("Smraiz Python time", time.time()-ct)
    return sol_order #, sol_set, sol_gains




class FASS(Strategy):
    
    """
    Implements FASS :footcite:`pmlr-v37-wei15` combines the uncertainty sampling 
    method with a submodular data subset selection framework to label a subset of data points to 
    train a classifier. Here the based on the ‘top_n’ parameter, ‘top_n*budget’ most uncertain 
    parameters are filtered. On these filtered points one of the submodular functions viz. 
    'facility_location' , 'feature_based', 'graph_cut', 'log_determinant', 'disparity_min', 'disparity_sum'
    is applied to get the final set of points.
    We select a subset :math:`F` of size :math:`\\beta` based on uncertainty sampling, such 
    that :math:`\\beta \\ge k`.
      
    Then select a subset :math:`S` by solving 
    
    .. math::
        \\max \\{f(S) \\text{ such that } |S| \\leq k, S \\subseteq F\\} 
    
    where :math:`k` is the is the `budget` and :math:`f` can be one of these functions - 
    'facility_location' , 'feature_based', 'graph_cut', 'log_determinant', 'disparity_min', 'disparity_sum'. 
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
        - **submod_args**: Parameters for the submodular selection as described in SubmodularSampling (dict, optional)
        - **uncertainty_measure**: Describes which measure of uncertainty should be used. This should be one of 'entropy', 'least_confidence', or 'margin' (string, optional)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(FASS, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'submod_args' in args:
            self.submod_args = args['submod_args']
        else:
            self.submod_args = {'submod': 'facility_location',
                                'metric': 'cosine'}
            self.args['submod_args'] = self.submod_args
        
        if 'uncertainty_measure' in args:
            self.uncertainty_measure = args['uncertainty_measure']
        else:
            self.uncertainty_measure = 'entropy'


    def select(self, budget, top_n=2):
        
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
        top_n: int, optional
            Number of slices of size budget to include in filtered subset
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	
        
        self.model.eval()
        
        # Determine if top_n * budget points can be drawn; otherwise, set filtered set size 
        # to be the size of the unlabeled_dataset
        filtered_set_size = min(budget * top_n, len(self.unlabeled_dataset))

        # Now, select the top 'filtered_set_size' most uncertain points using the 
        # specified measure of uncertainty (already implemented in strategies!)
        if self.uncertainty_measure == 'entropy':
            uncertainty_strategy = EntropySampling(self.labeled_dataset, self.unlabeled_dataset, self.model, self.target_classes, self.args)
        elif self.uncertainty_measure == 'least_confidence':
            uncertainty_strategy = LeastConfidenceSampling(self.labeled_dataset, self.unlabeled_dataset, self.model, self.target_classes, self.args)
        elif self.uncertainty_measure == 'margin':
            uncertainty_strategy = MarginSampling(self.labeled_dataset, self.unlabeled_dataset, self.model, self.target_classes, self.args)
        else:
            raise ValueError("uncertainty_measure must be one of 'entropy', 'least_confidence', or 'margin'")
        
        modular_scores = uncertainty_strategy.acquire_scores(self.unlabeled_dataset)
        
        # Now, use submodular selection to choose points from the filtered subset.
        # Ensure the representation type is in the submod_args dict.
        # if 'representation' not in self.submod_args:
        #     self.submod_args['representation'] = 'linear'
            
        # filtered_unlabeled_set = Subset(self.unlabeled_dataset, filtered_idxs)
        
        embeddings = self.get_embedding(self.unlabeled_dataset)
        # embeddings = embeddings.cpu().numpy()


        greedy_indices = get_facility_location_submodular_order(embeddings, modular_scores, 'rbf', budget, lam=5, c=0, divide_with=0, kernel_width=.1) #get_facility_location_submodular_order(embeddings, 'rbf', budget, divide_with=True, kernel_width=.1)

        # submodular_selection_strategy = SubmodularSampling(self.labeled_dataset, filtered_unlabeled_set, self.model, self.target_classes, self.args)
        # greedy_indices = submodular_selection_strategy.select(budget)
        
        # Lastly, map the indices of the filtered set to the indices of the unlabeled set
        # selected_indices = [filtered_idxs[x] for x in greedy_indices]
        
        return greedy_indices #selected_indices        