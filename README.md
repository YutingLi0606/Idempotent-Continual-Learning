# 📝 Idempotent continual learning

Official Repository for ICLR'26 Paper"Idempotent Experience Replay for Reliable Continual Learning"

IDER is a novel framework for continual learning based on the idempotent property, which mitigates catastrophic forgetting and improves prediction reliability. It is a simple and robust method that can be easily integrated into other state-of-the-art approaches.

## 🎉 News
- [x] **[2026.01.26]** Our paper has been accepted by ICLR 2026!
## Table of Content

* [1. Quick Start](#1-quick-start)
* [2. Reproduced Results](#2-reproduced-results)
* [3. Citation](#3-citation)
* [4. Acknowledgement](#4-acknowledgement)


## 1. Quick Start

### 1.1. Environment



Clone this repository and install the requirements. Our model can be learnt in a **single GPU RTX-4090 24G**

```bash
conda env create -f environment.yaml
conda activate icl
```
The code was tested on Python 3.10 and PyTorch 1.13.0.


### 1.2. Training
Train ResNet18 on S-CIFAR-100 using ER and ER+ID as baseline methods with 500 buffers. Run the following command:

```bash
bash run_para_cifar100.sh

```

## 2. Reproduced Results
After the paper has been accepted, we rerun everything to provide complete logs and checkpoints for our Table 1 in the paper. The example results are ResNet18 on S-CIFAR-100 using ER and ER+ID as baseline methods with 500 buffers and 0-4 seeds.
All results are stored in mlflow in thie repository. You can run mlflow ui server locally:
```bash
mlflow ui
```
And then go to http://127.0.0.1:5000/#/ in your brower to see all the results from the experiments we runned and exact hyperparameters used in each run.

Checkpoint

The checkpoints are saved under experiments folder.

## 3. Citation
If our project is helpful for your research, please consider citing :
```

@article{Li2024sureood,
    author    = {Li, Yang and Sha, Youyang and Wu, Shengliang and Li, Yuting and Yu, Xuanlong and Huang, Shihua and Cun, Xiaodong and Chen,Yingyi and Chen, Dexiong and Shen, Xi},
    title     = {SURE-OOD: Detecting OOD samples with SURE},
    month     = {September}
    year      = {2024},
}
```


## 4. Acknowledgement
This project is heavily based on [Mammoth](https://github.com/aimagelab/mammoth) and [weight-interpolation-cl](https://github.com/jedrzejkozal/weight-interpolation-cl). We sincerely appreciate the authors of the mentioned works for sharing such great library as open-source project.
