#!/bin/bash

# FOR CIFAR 10 

python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_derpp.json --seed 10 --gpu 0 --lr 0.02 --replay_size 40 --alpha .1 --beta 1  --fixed_epoch --csv_file results_cifar10.csv --n_epoch 40

python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_distill.json --seed 10 --gpu 0 --lr 0.02 --replay_size 40 --alpha .25 --beta .75  --fixed_epoch --csv_file results_cifar10.csv --n_epoch 40

python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_distilldiv.json --seed 10 --gpu 0 --lr 0.02 --replay_size 40 --alpha .25 --beta .75  --fixed_epoch --csv_file results_cifar10.csv --n_epoch 40 --kernel_width .1 --lam 1

python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_er.json --seed 10 --gpu 0 --lr 0.02 --replay_size 40 --alpha 0 --beta 0  --csv_file results_cifar10.csv --n_epoch 40 --fixed_epoch

python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_mir.json --seed 10 --gpu 0 --lr 0.02 --replay_size 40 --alpha 0 --beta 0  --csv_file results_cifar10.csv --n_epoch 40 --fixed_epoch

python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_nocl.json --seed 10 --gpu 0 --lr 0.02 --replay_size 0 --alpha 0 --beta 0  --csv_file results_cifar10.csv --n_epoch 100

python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_nocl.json --seed 10 --gpu 0 --lr 0.02 --replay_size 0 --alpha 0 --beta 0  --csv_file results_cifar10.csv --warm_start --n_epoch 100


