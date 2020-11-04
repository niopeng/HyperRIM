# HyperRIM: Hyper-Resolution Implicit Model

[Project Page][project] | [Paper][paper] | [Pre-trained Models][pretrain]

PyTorch implementation of HyperRIM model that upscales input images by a factor of 16x.

![Intro](../website/Teaser.jpg)

## Dependencies and Installation

- Python 3
- [PyTorch 1.4](https://pytorch.org)
- NVIDIA GPU + [CUDA 10.0](https://developer.nvidia.com/cuda-downloads)

To setup environment, please run the following command:
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
$ python train.py -opt options/train/train_butterfly_32to512.json
// Testing
$ python test.py -opt options/test/test_butterfly_32to512.json
```
Note: Training x16 HyperRIM model resolution from 32x32 to 512x512 needs 32G of GPU memory.


## Code Organization
The code is consisted of the following components:
- `/data`: Dataset/Dataloader definition and useful tools
- `/dciknn_cuda`: Fast k-Nearest Neighbour Search (DCI) interface
- `/models`: Defines HyperRIM model, architecture and [Leanred Perceptual Similarity (LPIPS)](https://github.com/richzhang/PerceptualSimilarity) loss
- `/options`: Training/Testing configurations
- `/utils`: Basic utlility functions, logger and progress bar
- `sampler.py`: Sampling process for conditional IMLE mentioned in our [paper][paper]
- `train.py`: Main training script
- `test.py`: Main testing script

[project]:https://niopeng.github.io/HyperRIM/
[paper]: https://arxiv.org/
[pretrain]: https://github.com/niopeng/HyperRIM/experiments/pretrained_models
[options]:https://github.com/niopeng/HyperRIM/codes/options
[data]:https://github.com/niopeng/HyperRIM/codes/data
