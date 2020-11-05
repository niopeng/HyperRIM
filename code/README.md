# HyperRIM: Hyper-Resolution Implicit Model

[Project Page][project] | [Paper][paper] | [Pre-trained Models][pretrain]

PyTorch implementation of HyperRIM: a conditional deep generative model that avoids mode collapse and can generate multiple outputs for the same input.
The model is trained with [Implicit Maximum Likelihood Estimation (IMLE)](https://arxiv.org/abs/1809.09087).
HyperRIM is able to:

- Increase the width and height of images by a factor of 16x
- Recover a plausible image from a badly compressed image
- Possibly do other things yet to be explored 😃

![Intro](../website/Teaser.jpg)

## Dependencies and Installation

- Python 3
- [PyTorch 1.4](https://pytorch.org)
- NVIDIA GPU + [CUDA 10.0](https://developer.nvidia.com/cuda-downloads)

To set up, please run the following command:
```sh
$ git clone --recursive https://github.com/niopeng/HyperRIM.git
$ cd code
$ pip install -r requirements.txt
$ cd dciknn_cuda
$ python setup.py install
```

## Training and Testing
Please run the following steps:
1. Prepare datasets. Details can be found [here][data].
2. Change config files located under [options][options].
3. [Optional] Load [pre-trained models][pretrain].
4. Run training/testing commands:
```sh
// Training
$ python train.py -opt options/train/[train_sr.json, train_decompression.json]
// Testing
$ python test.py -opt options/test/[test_sr.json, test_decompression.json]
```
Note: Training a HyperRIM model for 16x super-resolution requires 32GB of GPU memory, but testing only requires 16GB of GPU memory.


## Code Organization
The code consists of the following components:
- `data/`: Dataset/Dataloader definition and useful tools
- `dciknn_cuda/`: Fast k-Nearest Neighbour Search (DCI) interface
- `models/`: Defines HyperRIM model, architecture and [Leanred Perceptual Similarity (LPIPS)](https://github.com/richzhang/PerceptualSimilarity) loss
- `options/`: Training/Testing configurations
- `utils/`: Basic utlility functions, logger and progress bar
- `sampler.py`: Sampling process for conditional IMLE mentioned in our [paper][paper]
- `train.py`: Main training script
- `test.py`: Main testing script

[project]:https://niopeng.github.io/HyperRIM/
[paper]: https://arxiv.org/abs/2011.01926
[pretrain]: https://github.com/niopeng/HyperRIM/tree/main/experiments/pretrained_models
[options]:https://github.com/niopeng/HyperRIM/tree/main/code/options
[data]:https://github.com/niopeng/HyperRIM/tree/main/code/data
