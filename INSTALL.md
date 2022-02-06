# INSTALLATION GUIDE

This guide will walk you through the process of installing and using Hydra for the first time.

## Prerequisites
---

You must have the version of PyTorch installed that matches your system. You can download PyTorch here: https://pytorch.org/. The examples in this guide also use torchtext, but it is not necessary to use the framework.

The system is verified and tested for the latest versions of torch, torchvision, and torchtext.

However the examples are only verified for: torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 torchtext==0.9.1. To install these versions, use

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 torchtext==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

## Download
---

At the moment, Hydra is only available through this repository. Run `git clone https://github.com/knagrecha/mptorch`, then cd into the repo. 

Run `pip install -r requirements.txt`, then `pip install .`.


