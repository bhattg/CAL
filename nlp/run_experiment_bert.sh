#!/bin/bash
export LD_LIBRARY_PATH='.'
const=1
gpu=0
seed=10

fname="timing_config_cola_bert_entropysampling_nocl"
config_name="config_cola_bert_entropysampling_nocl"
CUDA_VISIBLE_DEVICES=$gpu python continual_train.py --bert  --config_path=continual_configs/$config_name.json --seed $seed --gpu 0 --csv_file "$fname.csv" 

fname="timing_config_cola_bert_entropysampling_wsnocl"
CUDA_VISIBLE_DEVICES=$gpu python continual_train.py --bert  --config_path=continual_configs/$config_name.json --seed $seed --gpu 0 --csv_file "$fname.csv" --warm_start

fname="timing_config_cola_bert_entropysampling_distilldiv"
config_name="config_cola_bert_entropysampling_distilldiv"

CUDA_VISIBLE_DEVICES=$gpu python continual_train.py --bert  --config_path=continual_configs/$config_name.json --seed $seed --gpu 0 --csv_file "$fname.csv" --fixed_epoch --alpha 0.5 --beta 0.5 --kernel_width 1 --lam 1

fname="timing_config_cola_bert_entropysampling_distill"
config_name="config_cola_bert_entropysampling_distill"

CUDA_VISIBLE_DEVICES=$gpu python continual_train.py --bert  --config_path=continual_configs/$config_name.json --seed $seed --gpu 0 --csv_file "$fname.csv" --fixed_epoch --alpha 0.75 --beta 0.25

fname="timing_config_cola_bert_entropysampling_derpp"
config_name="config_cola_bert_entropysampling_derpp"

CUDA_VISIBLE_DEVICES=$gpu python continual_train.py --bert  --config_path=continual_configs/$config_name.json --seed $seed --gpu 0 --csv_file "$fname.csv" --fixed_epoch --alpha 0.25 --beta 0.75

fname="timing_config_cola_bert_entropysampling_mir"
config_name="config_cola_bert_entropysampling_mir"

CUDA_VISIBLE_DEVICES=$gpu python continual_train.py --bert  --config_path=continual_configs/$config_name.json --seed $seed --gpu 0 --csv_file "$fname.csv" --fixed_epoch 


fname="timing_config_cola_bert_entropysampling_er"
config_name="config_cola_bert_entropysampling_er"

CUDA_VISIBLE_DEVICES=$gpu python continual_train.py --bert  --config_path=continual_configs/$config_name.json --seed 30 --gpu 0 --csv_file "$fname.csv" --fixed_epoch 

