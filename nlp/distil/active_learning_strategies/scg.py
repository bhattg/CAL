from .strategy import Strategy
import time 
import numpy as np
import os
from torch.utils.data import Subset
import torch
import submarine 
from distil.utils.utils import LabeledToUnlabeledDataset
# import submodlib



def get_transform(metric, divide_with=None, kernel_width=None):
    if metric == 'euclidean':
        s = "SIMEUCLIDEAN(exponent=1,TRANSPOSE())"
    elif metric == 'rbf':
        s =  f"RBFKERNEL(exponent_value=1,kernel_width={kernel_width}, divide_with={divide_with}, TRANSPOSE())"
    return "{"+s+"}"

def get_facility_location_submodular_order(U, L, metric, B, smraiz_path="smraiz", divide_with=None, kernel_width=None, scores=None):
    print("Computing submodular max")
    '''
    Args
    - U: np.array, 
    - X: np.array, shape [N, M], design matrix
    - B: int, number of points to select
    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    '''
    # print('Computing facility location submodular order...')


    X = torch.cat([L,U]).cpu()
    N = X.shape[0]
    contract_set = [i for i in range(L.shape[0])]





    if not(scores is None):
        scores = scores.cpu().numpy()

    ct = time.time()
    with open(f'./tmp/des-mat.npy', 'wb') as f:
        np.save(f, X)   

    # with open(f'./tmp/contract_set.npy', 'wb') as f:
    #     np.save(f, contract_set)

    contract_set = str(contract_set)

    if not (scores is None):
        with open(f'./tmp/modscore.npy', 'wb') as f:
            np.save(f, scores)

    # print(f"Matrix Store time {time.time()-ct}")

    ct = time.time()

    transform_kwd = get_transform(metric, divide_with, kernel_width)
    sgd = ""

    if not (scores is None):
        cmd = f'{smraiz_path} -sumsize {B} \
         {sgd} -flfile ./tmp/des-mat.npy -fltransformation "{transform_kwd}" -mofile ./tmp/modscore.npy -moresponsibility {lam} -pnpv -porder -ptime -nochecks -pnpysummary ./tmp/sum.npy -fl-set-contraction "{contract_set}"'
    else:
        cmd = f'time {smraiz_path} -sumsize {B} \
        {sgd} -flfile ./tmp/des-mat.npy -pnpv -porder -ptime -fltransformation "{transform_kwd}"  -nochecks -pnpysummary ./tmp/sum.npy -fl-set-contraction "{contract_set}" -set-deletion "{contract_set}"'

    # print(cmd)  
    ct = time.time() 
    os.system(cmd)
    print("Process Creation Time ",time.time()-ct)    
    order = np.load('./tmp/sum.npy')[:,0]

    order = [i - L.shape[0] for i in order]
    return order




class SCG(Strategy):
    
    """
    This strategy implements the Submodular Conditional Gain (SCG) selection paradigm discuss in the paper 
    SIMILAR: Submodular Information Measures Based Active Learning In Realistic Scenarios :footcite:`kothawade2021similar`. In this selection 
    paradigm, points from the unlabeled dataset are chosen in such a way that the submodular conditional gain 
    between this set of points and a provided private set is maximized. Doing so allows a practitioner to select 
    points from an unlabeled set that are dissimilar to points provided in the private set.
    
    These submodular conditional gain functions rely on formulating embeddings for the points in the unlabeled set 
    and the private set. Once these embeddings are formed, similarity kernels are formed from these 
    embeddings based on a similarity metric. Once these similarity kernels are formed, they are used in computing the value 
    of each submodular conditional gain function. Hence, common techniques for submodular maximization subject to a 
    cardinality constraint can be used, such as the naive greedy algorithm, the lazy greedy algorithm, and so forth.
    
    In this framework, we set the cardinality constraint to be the active learning selection budget; hence, a list of 
    indices with a total length less than or equal to this cardinality constraint will be returned. Depending on the 
    maximization configuration, one can ensure that the length of this list will be equal to the cardinality constraint.
    
    Currently, three submodular conditional gain functions are implemented: 'flcg', 'gccg', and 'logdetcg'. Each
    function is obtained by applying the definition of a submodular conditional gain function using common 
    submodular functions. For more information-theoretic discussion, consider referring to the paper Submodular Combinatorial 
    Information Measures with Applications in Machine Learning :footcite:`iyer2021submodular`.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled dataset to be used in this strategy. For the purposes of selection, the labeled dataset is not used, 
        but it is provided to fit the common framework of the Strategy superclass.
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled dataset to be used in this strategy. It is used in the selection process as described above.
        Importantly, the unlabeled dataset must return only a data Tensor; if indexing the unlabeled dataset returns a tuple of 
        more than one component, unexpected behavior will most likely occur.
    private_dataset: torch.utils.data.Dataset
        The private dataset to be used in this strategy. It is used in the selection process as described above. Notably, 
        the private dataset should be labeled; hence, indexing the query dataset should return a data/label pair. This is 
        done in this fashion to allow for gradient embeddings.
    net: torch.nn.Module
        The neural network model to use for embeddings and predictions. Notably, all embeddings typically come from extracted 
        features from this network or from gradient embeddings based on the loss, which can be based on hypothesized gradients 
        or on true gradients (depending on the availability of the label).
    nclasses: int
        The number of classes being predicted by the neural network.
    args: dict
        A dictionary containing many configurable settings for this strategy. Each key-value pair is described below:
            
            - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
            - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
            - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
            - **scg_function**: The submodular conditional gain function to use in optimization. Must be one of 'flcg', 'gccg', or 'logdetcg'.  (string)
            - **optimizer**: The optimizer to use for submodular maximization. Can be one of 'NaiveGreedy', 'StochasticGreedy', 'LazyGreedy' and 'LazierThanLazyGreedy'. (string, optional)
            - **metric**: The similarity metric to use for similarity kernel computation. This can be either 'cosine' or 'euclidean'. (string)
            - **nu**: A parameter that governs the hardness of the privacy constraint. (float)
            - **embedding_type**: The type of embedding to compute for similarity kernel computation. This can be either 'gradients' or 'features'. (string)
            - **gradType**: When 'embedding_type' is 'gradients', this defines the type of gradient to use. 'bias' creates gradients from the loss function with respect to the biases outputted by the model. 'linear' creates gradients from the loss function with respect to the last linear layer features. 'bias_linear' creates gradients from the loss function using both. (string)
            - **layer_name**: When 'embedding_type' is 'features', this defines the layer within the neural network that is used to extract feature embeddings. Namely, this argument must be the name of a module used in the forward() computation of the model. (string)
            - **stopIfZeroGain**: Controls if the optimizer should cease maximization if there is zero gain in the submodular objective. (bool)
            - **stopIfNegativeGain**: Controls if the optimizer should cease maximization if there is negative gain in the submodular objective. (bool)
            - **verbose**: Gives a more verbose output when calling select() when True. (bool)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}): #
        
        super(SCG, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        
        # self.private_dataset = private_dataset

    def select(self, budget):
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	

        self.model.eval()

        print(self.unlabeled_dataset, self.labeled_dataset)
        unlabeled_embedding = self.get_embedding(self.unlabeled_dataset)
        labeled_embedding = self.get_embedding(LabeledToUnlabeledDataset(self.labeled_dataset))
        print(labeled_embedding)
        order = get_facility_location_submodular_order(U=unlabeled_embedding, L=labeled_embedding, metric='euclidean', B=budget)
        return order




        # #Get hyperparameters from args dict
        # optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'NaiveGreedy'
        # metric = self.args['metric'] if 'metric' in self.args else 'cosine'
        # nu = self.args['nu'] if 'nu' in self.args else 1
        # gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        # stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        # stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        # verbose = self.args['verbose'] if 'verbose' in self.args else False
        # embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "gradients"
        # if(embedding_type=="features"):
        #     layer_name = self.args['layer_name'] if 'layer_name' in self.args else "avgpool"

        # #Compute Embeddings
        # if(embedding_type == "gradients"):
        #     unlabeled_data_embedding = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
        #     private_embedding = self.get_grad_embedding(self.private_dataset, False, gradType)
        # elif(embedding_type == "features"):
        #     unlabeled_data_embedding = self.get_feature_embedding(self.unlabeled_dataset, True, layer_name)
        #     private_embedding = self.get_feature_embedding(self.private_dataset, False, layer_name)
        # else:
        #     raise ValueError("Provided representation must be one of gradients or features")
        
        # #Compute image-image kernel
        # data_sijs = submodlib.helper.create_kernel(X=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        # #Compute private-private kernel
        # if(self.args['scg_function']=='logdetcg'):
        #     private_private_sijs = submodlib.helper.create_kernel(X=private_embedding.cpu().numpy(), metric=metric, method="sklearn")
        # #Compute image-private kernel
        # private_sijs = submodlib.helper.create_kernel(X=private_embedding.cpu().numpy(), X_rep=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        
        # if(self.args['scg_function']=='flcg'):
        #     obj = submodlib.FacilityLocationConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
        #                                                               num_privates=private_embedding.shape[0],  
        #                                                               data_sijs=data_sijs, 
        #                                                               private_sijs=private_sijs, 
        #                                                               privacyHardness=nu)
        
        # if(self.args['scg_function']=='gccg'):
        #     lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
        #     obj = submodlib.GraphCutConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
        #                                                               num_privates=private_embedding.shape[0],
        #                                                               lambdaVal=lambdaVal,  
        #                                                               data_sijs=data_sijs, 
        #                                                               private_sijs=private_sijs, 
        #                                                               privacyHardness=nu)
        # if(self.args['scg_function']=='logdetcg'):
        #     lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
        #     obj = submodlib.LogDeterminantConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
        #                                                               num_privates=private_embedding.shape[0],
        #                                                               lambdaVal=lambdaVal,  
        #                                                               data_sijs=data_sijs, 
        #                                                               private_sijs=private_sijs,
        #                                                               private_private_sijs=private_private_sijs, 
        #                                                               privacyHardness=nu)

        # greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
        #                       stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices