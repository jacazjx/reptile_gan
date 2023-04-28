# TwinGAN: 

PyTorch implementation of OpenAI's Reptile algorithm for supervised learning.

Currently, it runs on Omniglot but not yet on MiniImagenet.

The code  has not been tested extensively. Contributions and feedback are more than welcome!

## Omniglot meta-learning dataset

There is already an Omniglot dataset class in torchvision, however it seems to be more adapted for supervised-learning
than few-shot learning.

The `omniglot.py` provides a way to sample K-shot N-way base-tasks from Omniglot, 
and various utilities to split meta-training sets as well as base-tasks.

## Features

- [x] Peer-to-Peer Federated Learning.
- [x] Meta Learning for Generator. 
- [x] Knowledge Distillition for Fine-tuning .

## How to train on Omniglot

Download the two parts of the Omniglot dataset:
- https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
- https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip

Create a `omniglot/` folder in the repo, unzip and merge the two files to have the following folder structure:
```
./train_omniglot.py
...
./omniglot/Alphabet_of_the_Magi/
./omniglot/Angelic/
./omniglot/Anglo-Saxon_Futhorc/
...
./omniglot/ULOG/
```

Now start training with
```
python train_omniglot.py log --cuda 0 $HYPERPARAMETERS  # with CPU
python train_omniglot.py log $HYPERPARAMETERS  # with CUDA
```
where $HYPERPARAMETERS depends on your task and hyperparameters.

Behavior:
- If no checkpoints are found in `log/`, this will create a `log/` folder to store tensorboard information and checkpoints.
- If checkpoints are found in `log/`, this will resume from the last checkpoint.

Training can be interrupted at any time with `^C`, and resumed from the last checkpoint by re-running the same command.

## Omniglot Hyperparameters

The following set of hyperparameters work decently. 
They are taken from the OpenAI implementation but are adapted slightly
for `meta-batch=1`.

<img src="https://github.com/gabrielhuang/reptile-pytorch/raw/master/plots/omniglot_train.png" width="400">
<img src="https://github.com/gabrielhuang/reptile-pytorch/raw/master/plots/omniglot_val.png" width="400">

## MNIST

FeGAN
```bash
python md_fe_gan.py --model fegan -N 10 --check_every 100
```

MDGAN 1-Inner_Loop Conditional GAN under IID and NonIID
```bash
python md_fe_gan.py --model mdgan -N 10 --meta_epochs 1  --check_every 100 
```

TwinGAN 5-Task, 4-Mini_batch, 5-Inner_Loop  
```bash
python dismeta.py -N 10 --num_tasks 5 --meta_epochs 5 --batch 20  --check_every 100 --condition --meta --ln
```

## EMNIST

FeGAN
```bash
python md_fe_gan.py --model fegan -N 10 --dataset emnist --niid  --check_every 500
```

MDGAN 1-Inner_Loop Conditional GAN under NonIID
```bash
python md_fe_gan.py --model mdgan -N 10 --dataset emnist --niid --meta_epochs 1
```

TwinGAN 5-Task, 4-Mini_batch, 5-Inner_Loop
```bash
python dismeta.py -N 10 --num_tasks 5 --dataset emnist --niid --meta_epochs 5 --batch 20  --check_every 100 --condition --meta --ln
```

## References

- [Original Paper](https://arxiv.org/abs/1803.02999): Alex Nichol, Joshua Achiam, John Schulman. "On First-Order Meta-Learning Algorithms".
- [OpenAI blog post](https://blog.openai.com/reptile/). 
Check it out, they have an online demo running entirely in Javascript!
- Original code in Tensorflow: https://github.com/openai/supervised-reptile
