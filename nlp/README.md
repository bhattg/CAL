# Accelerating Active Learning Using Continual Learning (TMLR'23, Dec Edition) 

This is the codebase for our TMLR paper -- [Accelerating Batch Active Learning Using Continual Learning Techniques](https://openreview.net/forum?id=T55dLSgsEf), also referred to as Continual Active Learning (CAL). 


```
A major problem with Active Learning (AL) is high training costs since models are typically retrained from scratch after every query round. We start by demonstrating that standard AL on neural networks with warm starting fails, both to accelerate training and to avoid catastrophic forgetting when using fine-tuning over AL query rounds.  We then develop a new class of techniques, circumventing this problem, by biasing further training towards previously labeled sets. We accomplish this by employing existing, and developing novel, replay-based Continual Learning (CL) algorithms that are effective at quickly learning the new without forgetting the old, especially when data comes from an evolving distribution. We call this paradigm \textit{"Continual Active Learning" (CAL)}.  We show CAL achieves significant speedups using a plethora of replay schemes that use model distillation and that select diverse/uncertain points from the history. We conduct experiments across many data domains, including natural language, vision, medical imaging, and computational biology, each with different neural architectures and dataset sizes. CAL consistently provides a 
3x reduction in training time, while retaining performance and out-of-distribution robustness, showing its wide applicability.
```

On the implementation of Active Learning strategies, we use [DISTIL framework](https://github.com/decile-team/distil). After meeting wiith the the necessary requirements for DISTIL Framework, please go ahead and run the following to install dependencies, followed by necessary imports for submodular maximization. 

`pip install -r requirements/requirements.txt`

## Implementing custom Continual Active Learning Algorithm! 

Implementing a custom CAL algorithm requires two things -- writing strategy file similar to `distil/utils/continual_learning/distill.py`, followed by a config file in `continual_configs/`. Config `.json` file that provides a list of configuration will be used for the Continual Learning algorithm. For example, refer to the following - 

```
{
        "model": {
                "architecture": "bert",
                "target_classes": 2
        },
        "train_parameters": {
                "lr": 5e-5,
                "batch_size": 25,
                "n_epoch": 5,
                "max_accuracy": 0.99,
                "isreset": false,
                "islogs":  true,
                "isverbose":  true,
                "logs_location": "./logs.txt",
                "optimizer":  "adam"
        },

        "active_learning":{
                "strategy": "entropy_sampling",
                "budget": 50,
                "rounds": 20,
                "initial_points": 50,

                "strategy_args":{
                        "batch_size" : 20,
                        "bert" : 1,
                        "lr":0.01
                }
        },

        "continual_learning":{
                "replay_strategy": "distill",
                "alpha": 1,
                "beta": 1,
                "replay_size": 25,
                "first_n_epoch": 5,
                "n_epoch": 5,
                "target_classes": 2
        },

        "dataset":{
                "name":"cola"
        }
}

```

Lastly, import the new strategy in `distil/utils/continual_learning/__init__.py`. Note that if a new architecture is implemented, then please make sure to add that in `distil/utils/models/` and import them in `distil/utils/models/__init__.py`

## Running experiments in the paper

As an example, to run the code on COLA with BERT backbone, using Distill (CAL-SDS) as the CAL algorithm, one can use the following -- 

```python continual_train.py --config_path=continual_configs/config_cifar10_resnet_entropysampling_distill.json --seed 10 --gpu 0 --lr 0.02 --replay_size 40 --alpha .25 --beta .75  --fixed_epoch --csv_file results_cifar10.csv --n_epoch 40```

For our experiments in paper, please consider running the following scripts (scripts can be modified similarly for the Amazon polarity dataset with VDCNN architecture)

`./run_experiments_bert.sh`

**Note that, for the Vision related experiments, please refer to `../vision/`**

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