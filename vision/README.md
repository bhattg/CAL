# Acceleratiing Active Learning Using Continual Learning 

This is the codebase for our TMLR paper 



We directly use DISTIL framework from https://github.com/decile-team/distil 


Install all dependencies by running:
`pip3 install -r requirements/requirements.txt`

First allow submarine to be imported by running the following command:
`export LD_LIBRARY_PATH=.`


Run all cifar-10 entropy sampling experiments with
`./run_experiments_cifar.sh`

Run all MedMNIST entropy sampling experiments with
`./run_experiments_cifar.sh`


For FMNIST/Amazon Polarity, we provide exmaple .json files, which can be changed accordingly (other dataset and other acquisition function) to run the experiments as desired.  

For BERT experiment, please see the BERT EXPEREIMENTS folder and `BERT\EXPEREIMENTS/run_experiment_bert.sh`. All the BERT experiments should be launched from that folder. 




If you find this work useful, please consider citing us with - 

```
@article{
das2023accelerating,
title={Accelerating Batch Active Learning Using Continual Learning Techniques},
author={Arnav Mohanty Das and Gantavya Bhatt and Megh Manoj Bhalerao and Vianne R. Gao and Rui Yang and Jeff Bilmes},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=T55dLSgsEf},
note={}
}
```