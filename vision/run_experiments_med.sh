#!/bin/bash
# FOR MEDMNIST 

## for entropy and AL

python continual_train.py --config_path=continual_configs/config_dermamnist_resnet_entropy_nocl.json --seed 10 --gpu 0 --lr 0.001 --replay_size 0 --alpha 0 --beta 0 --csv_file dermamnist_resnet_entropy_nocl.csv --n_epoch 100

# for entropy and WS AL
python continual_train.py --config_path=continual_configs/config_dermamnist_resnet_entropy_nocl.json --seed 10 --gpu 0 --lr 0.001 --replay_size 0 --alpha 0 --beta 0 --csv_file dermamnist_resnet_entropy_nocl_warmstart.csv --warm_start --n_epoch 100 


num_epoch=20
rsize=128
## for entropy and ER
 python continual_train.py --config_path=continual_configs/config_dermamnist_resnet_entropy_ER.json --seed 10 --gpu 0 --lr 0.001 --replay_size $rsize --alpha 0 --beta 0 --csv_file dermamnist_resnet_entropy_ER.csv --n_epoch $num_epoch --fixed_epoch


## for entropy and MIR
 python continual_train.py --config_path=continual_configs/config_dermamnist_resnet_entropy_MIR.json --seed 10 --gpu 0 --lr 0.001 --replay_size $rsize --alpha 0 --beta 0 --csv_file dermamnist_resnet_entropy_MIR.csv --n_epoch $num_epoch --fixed_epoch

## for entropy and DERPP
python continual_train.py --config_path=continual_configs/config_dermamnist_resnet_entropy_derpp.json --seed 10 --gpu 0 --lr 0.001 --replay_size $rsize --alpha 0.1 --beta 1 --csv_file dermamnist_resnet_entropy_derpp.csv --n_epoch $num_epoch --fixed_epoch

## for entropy and SD
python continual_train.py --config_path=continual_configs/config_dermamnist_resnet_entropy_distill.json --seed 10 --gpu 0 --lr 0.001 --replay_size $rsize --alpha 0.5 --beta 0.5 --csv_file dermamnist_resnet_entropy_distill.csv --n_epoch $num_epoch --fixed_epoch

## for entropy and SDS2

 python continual_train.py --config_path=continual_configs/config_dermamnist_resnet_entropy_distilldiv.json --seed 10 --gpu 0 --lr 0.001 --replay_size $rsize --alpha 0.5 --beta 0.5 --csv_file dermamnist_resnet_entropy_distilldiv.csv --n_epoch $num_epoch --lam 10 --kernel_width 0.1 --fixed_epoch
