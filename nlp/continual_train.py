import numpy as np
import sys
# from sklearn.preprocessing import StandardScaler
import argparse
sys.path.append('./')
import torch
from torch.utils.data import Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from distil.utils.models.cola_model import BERT_COLA
from distil.utils.models.resnet import ResNet18
from distil.utils.models.nlp_models import MODELS
from distil.utils.dataset import MedMNISTDataset
# from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, LeastConfidenceSampling, \
#                                         MarginSampling, CoreSet, AdversarialBIM, AdversarialDeepFool, KMeansSampling, \
#                                         BALDDropout, FASS

from distil.active_learning_strategies import EntropySampling, RandomSampling, BADGE, FASS, GLISTER, SCG

from distil.utils.models.simple_net import TwoLayerNet
from distil.utils.continual_learning import *
from distil.utils.config_helper import read_config_file
from distil.utils.utils import LabeledToUnlabeledDataset
import time
import pickle
import random
import csv
import os 
import wandb
from nlp_datasets import *


# print("imported")

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    print("Setting seeds ...... \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True


class TrainClassifier:
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = read_config_file(config_file)

    def getModel(self, model_config):

        if model_config['architecture'] == 'bert':
            net = BERT_COLA()

        if model_config['architecture'] == 'resnet18':

            if ('target_classes' in model_config) and ('channel' in model_config):
                net = ResNet18(num_classes = model_config['target_classes'], channels = model_config['channel'])
            elif 'target_classes' in model_config:
                net = ResNet18(num_classes = model_config['target_classes'])
            else:
                net = ResNet18()
        
        elif model_config['architecture'] == 'two_layer_net':
            net = TwoLayerNet(model_config['input_dim'], model_config['target_classes'], model_config['hidden_units_1'])
        elif model_config['architecture'] == 'vdcnn9-maxpool':
            net = MODELS['vdcnn9-maxpool']()

        return net

    def libsvm_file_load(self, path,dim, save_data=False):

        data = []
        target = []
        with open(path) as fp:
           line = fp.readline()
           while line:
            temp = [i for i in line.strip().split(" ")]
            target.append(int(float(temp[0]))) # Class Number. # Not assumed to be in (0, K-1)
            temp_data = [0]*dim
            
            for i in temp[1:]:
                ind,val = i.split(':')
                temp_data[int(ind)-1] = float(val)
            data.append(temp_data)
            line = fp.readline()
        X_data = np.array(data,dtype=np.float32)
        Y_label = np.array(target)
        if save_data:
            # Save the numpy files to the folder where they come from
            data_np_path = path + '.data.npy'
            target_np_path = path + '.label.npy'
            np.save(data_np_path, X_data)
            np.save(target_np_path, Y_label)

        return (X_data, Y_label)

    def getData(self, data_config):
        
        # print(data_config)
        if data_config['name'] == 'amazon-polarity-review':
            train_dataset, train_transform = create_dataset('amazon_review_polarity', './', train=True)
            test_dataset, _ = create_dataset('amazon_review_polarity', './', train=False)

        if data_config['name'] == 'cifar10':

            download_path = './downloaded_data/'

            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            # train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            train_dataset = datasets.CIFAR10(download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
            test_dataset = datasets.CIFAR10(download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        elif data_config['name'] == 'mnist':

            download_path = './downloaded_data/'
            image_dim=28
            train_transform = transforms.Compose([transforms.RandomCrop(image_dim, padding=4), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            test_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
            test_dataset = datasets.MNIST(download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        elif data_config['name'] == 'fmnist':
            
            download_path = './downloaded_data/'
            
            train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Use mean/std of MNIST

            train_dataset = datasets.FashionMNIST(download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
            test_dataset = datasets.FashionMNIST(download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        elif data_config['name'] == 'svhn':
            
            download_path = './downloaded_data/'
            
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

            train_dataset = datasets.SVHN(download_path, download=True, split='train', transform=train_transform, target_transform=torch.tensor)
            test_dataset = datasets.SVHN(download_path, download=True, split='test', transform=test_transform, target_transform=torch.tensor) 

        elif data_config['name'] == 'cifar100':
            
            download_path = './downloaded_data/'
            
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

            train_dataset = datasets.CIFAR100(download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
            test_dataset = datasets.CIFAR100(download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        elif data_config['name'] == 'stl10':
            
            download_path = './downloaded_data/'
            
            train_transform = transforms.Compose([transforms.RandomCrop(96, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std

            train_dataset = datasets.STL10(download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
            test_dataset = datasets.STL10(download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        elif data_config['name'] == 'satimage':

            trn_file = '../distil_datasets/satimage/satimage.scale.trn'
            tst_file = '../distil_datasets/satimage/satimage.scale.tst'
            data_dims = 36

            X, y = self.libsvm_file_load(trn_file, dim=data_dims)
            X_test, y_test = self.libsvm_file_load(tst_file, dim=data_dims)

            y -= 1  # First Class should be zero
            y_test -= 1  # First Class should be zero

            sc = StandardScaler()
            X = sc.fit_transform(X)
            X_test = sc.transform(X_test)
            
            train_dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
            test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

        elif data_config['name'] == 'ijcnn1':
            
            trn_file = '../datasets/ijcnn1/ijcnn1.trn'
            tst_file = '../datasets/ijcnn1/ijcnn1.tst'
            data_dims = 22
            
            X, y = self.libsvm_file_load(trn_file, dim=data_dims)
            X_test, y_test = self.libsvm_file_load(tst_file, dim=data_dims) 

            # The class labels are (-1,1). Make them to (0,1)
            y[y < 0] = 0
            y_test[y_test < 0] = 0    

            sc = StandardScaler()
            X = sc.fit_transform(X)
            X_test = sc.transform(X_test)
            
            train_dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
            test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

        elif data_config['name'] == 'dermamnist':
            data_medmnist = np.load("../data/dermamnist.npz")
            train_images, train_labels, val_images, val_labels, test_images, test_labels = data_medmnist["train_images"].astype(np.float32), data_medmnist["train_labels"][:,0].astype(np.longlong), data_medmnist["val_images"].astype(np.float32), data_medmnist["val_labels"][:,0].astype(np.longlong), data_medmnist["test_images"].astype(np.float32), data_medmnist["test_labels"][:,0].astype(np.longlong)
            mean, std = self.get_mean_std(train_images, train_labels)
            #for img in range(train_images.shape[0]):
            #    print(type(train_images[img,:,:,:].shape))
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=mean)])

            train_dataset = MedMNISTDataset(tensors=(train_images, train_labels), transform = data_transform)
            test_dataset = MedMNISTDataset(tensors=(test_images, test_labels), transform=data_transform)
            num_classes = len(np.unique(train_labels))
            print("num class in", data_config['name'], "is", num_classes)

        return train_dataset, test_dataset

    def write_logs(self, logs, save_location, rd):
      
        file_path = save_location
        with open(file_path, 'a') as f:
            f.write('---------------------\n')
            f.write('Round '+str(rd)+'\n')
            f.write('---------------------\n')
            for key, val in logs.items():
                if key == 'Training':
                    f.write(str(key)+ '\n')
                    for epoch in val:
                        f.write(str(epoch)+'\n')       
                else:
                    f.write(str(key) + ' - '+ str(val) +'\n')

    def train_classifier(self, args=None):
        
        if args.wandb:
            wandb.init()


        net = self.getModel(self.config['model'])
        full_train_dataset, test_dataset = self.getData(self.config['dataset'])
        selected_strat = self.config['active_learning']['strategy']
        budget = self.config['active_learning']['budget']
        start = self.config['active_learning']['initial_points']
        n_rounds = self.config['active_learning']['rounds']
        nclasses = self.config['model']['target_classes']
        strategy_args = self.config['active_learning']['strategy_args'] 
        val_set_size = 2000
        continual_strat = self.config['continual_learning']['replay_strategy']
        pyramid = []

        # print(n_rounds)

        t0 = time.time()
        nSamps = len(full_train_dataset)
        np.random.seed(42)
        # start_idxs = np.random.choice(nSamps, size=start + val_set_size, replace=False)
        # train_dataset = Subset(full_train_dataset, start_idxs[:-val_set_size])
        # val_set = Subset(full_train_dataset, start_idxs[-val_set_size:])
        start_idxs = np.random.choice(nSamps, size=start, replace=False)
        train_dataset = Subset(full_train_dataset, start_idxs)
        val_set = train_dataset
        unlabeled_dataset = Subset(full_train_dataset, list(set(range(len(full_train_dataset))) -  set(start_idxs)))
        
        if 'islogs' in self.config['train_parameters']:
            islogs = self.config['train_parameters']['islogs']
            save_location = self.config['train_parameters']['logs_location']
        else:
            islogs = False
               
        logs = {}
        logs['CL Method'] = continual_strat
        


        # Continual learning strat
        if continual_strat == 'er':
            dt = ER(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'mir':
            dt = MIR(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'no_replay':
            dt = No_Replay(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'der' or continual_strat == 'derpp':
            dt = DER(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'diver':
            dt = DivER(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'no_cl':
            dt = No_CL(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'divder':
            dt = DivDER(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'distill':
            dt = Distill(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        elif continual_strat == 'distill_div' or 'distilldiv':
            dt = DistillDiv(train_dataset, val_set, net, self.config['train_parameters'], self.config['continual_learning'], self.config['active_learning'],args)
        else:
            raise IOError('Enter Valid CL Strategy')


        # Active learning strat
        if selected_strat == 'glister':
            strategy = GLISTER(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'badge':
            strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'entropy_sampling':
            strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'margin_sampling':
            strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'least_confidence':
            strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'random_sampling':
            strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'fass':
            strategy = FASS(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'scg':
            strategy = SCG(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        else:
            raise IOError('Enter Valid Strategy!')      
   
        times = {}

        tic = time.time()
        # TRAIN ON INITIAL LABELLED POOL
        if islogs:
            clf, train_logs= dt.train(first_round=True)
        else:
            clf = dt.train(first_round=True)
        strategy.update_model(clf)
        dt.update_buffer()
        toc = time.time()

        acc = [''] * n_rounds
        acc[0] = str(dt.get_acc_on_set(test_dataset))
   
        if islogs:
            logs['Task 0'] = str(round(np.float32(acc[0]), 4))
            times['Time 0'] = str(np.round(toc - tic, 2))
        
        print('Testing Accuracy:', acc[0])
        
        print('***************************')
        print('Starting Training..')
        print('***************************')
        ##User Controlled Loop

        for rd in range(1, n_rounds):
            print('***************************')
            print('Round', rd)
            print('***************************')        
            # logs = {}
            # t0 = time.time()
            idx = strategy.select(budget)
            # t1 = time.time()

            #Adding new points to training set
            # train_dataset = ConcatDataset([train_dataset, Subset(unlabeled_dataset, idx)])
            cur_dataset = Subset(unlabeled_dataset, idx)
            remain_idx = list(set(range(len(unlabeled_dataset))) - set(idx))
            unlabeled_dataset = Subset(unlabeled_dataset, remain_idx)
   
            dt.update_task(cur_dataset)

            print('Total training points in this round', dt.buffer.get_size() + len(cur_dataset))
            strategy.update_data(ConcatDataset([train_dataset, cur_dataset]), LabeledToUnlabeledDataset(unlabeled_dataset))
   
            tic = time.time()
            if islogs:
                if rd == n_rounds - 1 and args.dss:
                    clf, train_logs = dt.train(last_round=True)
                else:
                    clf, train_logs = dt.train()
            else:
                if rd == n_rounds - 1 and args.dss:
                    clf = dt.train(last_round=True)
                else:
                    clf = dt.train()
            toc = time.time()
            
            strategy.update_model(clf)
            acc[rd] = str(np.round(dt.get_acc_on_set(test_dataset), 4))
            print('Testing Accuracy:', acc[rd])
   
            if islogs:
                # logs['Training Points'] = len(train_dataset)
                # logs['Test Accuracy'] =  str(round(np.float32(acc[rd])*100, 2))
                # logs['Selection Time'] = str(t1 - t0)
                # logs['Training Time'] = str(t2 - t1) 
                logs['Task ' + str(rd)] = acc[rd]
                times['Time ' + str(rd)] = str(np.round((toc - tic) + np.float32(times['Time ' + str(rd-1)]), 2))
                # logs['Training'] = train_logs
                # self.write_logs(logs, save_location, rd)


            dt.update_buffer() 
            if args.pyramid:
                buffer_accs = dt.evaluate_buffer()
                pyramid.append(buffer_accs)
                print(buffer_accs)

        t1 = time.time()
        print('Training Completed!')
        with open('./final_model.pkl', 'wb') as save_file:
            pickle.dump(clf.state_dict(), save_file)
        print('Model Saved!')


        for k,v in times.items():
            logs[k] = v

        logs['Runtime'] = np.round(np.float32(t1 - t0), 2)
        logs['Seed'] = int(args.seed)
        logs['Alpha'] = args.alpha
        logs['Beta'] = args.beta
        logs['KW'] = args.kernel_width
        logs['Lambda'] = args.lam
        logs['Replay Size'] = args.replay_size
        logs['LR'] = args.lr 



        if args.pyramid:
            print(pyramid)

        if args.csv_file:
            filename = args.csv_file
            writeheader = False
            if not os.path.exists(filename):
                writeheader = True

            with open(filename, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=logs.keys())
                if writeheader:
                    writer.writeheader()
                writer.writerow(logs)



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True)
    parser.add_argument('--config_path', required=True, help="Path to the config file")
    parser.add_argument('--lr', required=False, help="CL lr")
    parser.add_argument('--wandb', action='store_true', help='Toggle wandb')
    parser.add_argument('--gpu', required=False, help='which GPU to use')
    parser.add_argument('--replay_size', required=False, help='how many samples to draw from buffer')
    parser.add_argument('--bert', action='store_true', help='when training bert models')
    parser.add_argument('--warm_start', action='store_true', help='warm start between training rounds')
    parser.add_argument('--fixed_epoch', action='store_true', help='set to true to disable early stopping')
    parser.add_argument('--csv_file', required=False, help="Which csv file to write experiment results")
    parser.add_argument('--dss', action='store_true', help="Treat AL as data subset selection (Reset and apply full training on final round)")
    parser.add_argument('--partial_ws', action='store_true', help="Apply shrink and perturb initialization")
    parser.add_argument('--n_epoch', required=False, help="Number of epochs in CL rounds")
    parser.add_argument('--pyramid', action='store_true', help="Print performance on each task")

    # DERPP
    parser.add_argument('--alpha', required=False, help="DERPP parameter")
    parser.add_argument('--beta', required=False, help="DERPP parameter")
    
    # DIVER
    parser.add_argument('--kernel_width', required=False, help="rbf parameter")
    parser.add_argument('--lam', required=False, help='Tradeoff parameter between uncertainty and diversity')
    args = parser.parse_args()

    set_random_seed(int(args.seed))
    tc = TrainClassifier(args.config_path)
    tc.train_classifier(args)


# tc = TrainClassifier('./configs/config_2lnet_satimage_randomsampling.json')
# # tc = TrainClassifier('./configs/config_cifar10_marginsampling.json')
# tc.train_classifier()