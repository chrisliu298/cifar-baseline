# CIFAR-10/100 Baseline

This repository contains code to train CNN architectures on [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) using [PyTorch Lightning](https://www.pytorchlightning.ai/).

The setup I used mainly follows [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610). The learning rates and weight decay factors are shown below. In the paper, the author reported `{"lr": 0.05, "wd": 0.001}` and `{"lr": 3e-4, "wd": 1}` the best for SGD and AdamW, respectively.

```json
{
   "adamw":{
      "lr":[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
      "wd":[0, 0.1, 0.3, 1, 3]
   },
   "sgd":{
      "lr":[0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
      "wd":[0, 0.0003, 0.001, 0.003]
   }
}
```

## Acknowledgment

- [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- [szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)
