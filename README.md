



# 📝 Idempotent continual learning

Official Repository for ICLR'26 Paper"Idempotent Experience Replay for Reliable Continual Learning"



## 🎉 News
- [x] **[2026.01.26]** Our paper has been accepted by ICLR 2026!
## Table of Content

* [1. Installation](#1-installation)
* [4. Quick Start](#4-quick-start)
* [5. Citation](#5-citation)
* [6. Acknowledgement](#6-acknowledgement)



## 1. Installation

### 1.1. Environment


Our model can be learnt in a **single GPU RTX-4090 24G**

```bash
conda env create -f environment.yaml
conda activate icl
```

The code was tested on Python 3.10 and PyTorch 1.13.0.


### 3.2. Datasets
#### 3.2.1 CIFAR and Tiny-ImageNet
* Using **CIFAR10, CIFAR100 and Tiny-ImageNet** for failure prediction (also known as misclassification detection).
* We keep **10%** of training samples as a validation dataset for failure prediction. 
* Download datasets to ./data/ and split into train/val/test.
Take CIFAR10 for an example:
```
cd data
bash download_cifar.sh
```
The structure of the file should be:
```
./data/CIFAR10/
├── train
├── val
└── test
```


## 5. Citation
If our project is helpful for your research, please consider citing :
```
@InProceedings{Li_2024_CVPR,
    author    = {Li, Yuting and Chen, Yingyi and Yu, Xuanlong and Chen, Dexiong and Shen, Xi},
    title     = {SURE: SUrvey REcipes for building reliable and robust deep networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17500-17510}
}

@article{Li2024sureood,
    author    = {Li, Yang and Sha, Youyang and Wu, Shengliang and Li, Yuting and Yu, Xuanlong and Huang, Shihua and Cun, Xiaodong and Chen,Yingyi and Chen, Dexiong and Shen, Xi},
    title     = {SURE-OOD: Detecting OOD samples with SURE},
    month     = {September}
    year      = {2024},
}
```



## 6. Acknowledgement
We refer to codes from [FMFP](https://github.com/Impression2805/FMFP) and [OpenMix](https://github.com/Impression2805/OpenMix). Thanks for their awesome works.


