
<div align="center">

# Idempotent Experience Replay for Reliable Continual Learning

<a href="ARXIV_LINK"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
<a href="PROJECT_PAGE_LINK"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>

IDER is a novel framework for continual learning based on the idempotent property, which mitigates catastrophic forgetting and improves prediction reliability. It is a simple and robust method that can be easily integrated into other state-of-the-art approaches.

<hr/>

<p>
Zhanwang Liu<sup>1</sup>, Yuting Li<sup>1</sup>, Haoyuan Gao<sup>1</sup>, Yexin Li<sup>4</sup>, Linghe Kong<sup>1</sup>, Lichao Sun<sup>3</sup>, Weiran Huang<sup>1,2,4</sup>
</p>

<p>
<sup>1</sup> School of Computer Science, Shanghai Jiao Tong University<br/>
<sup>2</sup> Shanghai Innovation Institute<br/>
<sup>3</sup> Lehigh University<br/>
<sup>4</sup> State Key Laboratory of General Artificial Intelligence, BIGAI
</p>
</div>

## 🎉 News
- [x] **[2026.01.26]** Our paper has been accepted by ICLR 2026!
## Table of Content

* [1. Quick Start](#1-quick-start)
* [2. Reproduced Results](#2-reproduced-results)
* [3. Tools](#3-tools)
* [4. Citation](#4-citation)
* [5. Acknowledgement](#5-acknowledgement)


## 1. Quick Start

### 1.1. Environment



Clone this repository and install the requirements. Our model can be learnt in a **single GPU RTX-4090 24G**

```bash
conda env create -f environment.yaml
conda activate icl
```
The code was tested on Python 3.10 and PyTorch 1.13.0.


### 1.2. Training

Train and evaluate ResNet18 on different datasets using ER and ER+ID with different buffers. Run the following command:

<details>
<summary><b>CIFAR-10</b></summary>
    
```bash
bash run_para_cifar10.sh
``` 
</details>
<details> <summary><b>CIFAR-100</b></summary>
    
```bash
bash run_para_cifar100.sh
```
</details>
<details> <summary><b>TinyImageNet</b></summary>
    
```bash
bash run_para_tinyimg.sh
```
</details>

## 2. Reproduced Results
After the paper has been accepted, we rerun everything to provide complete logs and checkpoints for our Table 1 in the paper. The example results are ResNet18 on different datasets using ER and ER+ID as baseline methods with different buffers and 0-4 seeds.
| **Dataset** | **Buffer** | **Method** | **Forgetting(⬇️)** | **TIL(⬆️)** | **CIL(⬆️)** | **Checkpoint** |
|---|---:|---|---:|---:|---:|---|
| CIFAR-10 | 200 | ER | 59.71 ± 2.62 | 91.48 ± 0.93 | 48.89 ± 2.19 | - |
|  |  | ER+ID | 16.89 ± 2.26 | 95.87 ± 0.36 | 70.68 ± 1.10 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar10/buffer_200/erid_seed_0.pth) |
|  | 500 | ER | 44.75 ± 2.94  | 93.38 ± 0.36 | 60.62 ± 2.46 | - |
|  |  | ER+ID | 11.59 ± 2.13 | 96.20 ± 0.40 | 75.52 ± 1.35 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar10/buffer_500/erid_seed_0.pth) |
| CIFAR-100 | 500 | ER | 73.81 ± 0.42  | 73.98 ± 1.15 | 21.28 ± 1.08 | - |
|  |  | ER+ID | 32.27 ± 1.96 | 83.30 ± 0.41 | 45.21 ± 1.20 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar100/buffer_500/erid_seed_0.pth) |
|  | 2000 | ER | 54.52 ± 0.62 | 81.62 ± 0.95 | 37.93 ± 0.76 | - |
|  |  | ER+ID | 18.76 ± 1.52 | 86.54 ± 0.34  | 56.30 ± 0.50  | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar100/buffer_2000/erid_seed_0.pth) |
| Tiny-ImageNet | 4000 | ER | 56.89 ± 0.74 | 66.68 ± 0.47 | 25.20 ± 0.70 | - |
|  |  | ER+ID | 21.62 ± 1.67 | 74.56 ± 0.55 | 43.25 ± 1.26 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-tinyimg/buffer_4000/erid_seed_0.pth) |



Checkpoint

The checkpoints are saved under experiments folder.


## 3. Tools
<details>
<summary><b>mlflow visulization</b></summary>
    
1. Setup
    
```bash
pip install mlflow
```

2. All results are stored in mlflow under the repository mlruns. You can run mlflow ui server locally:
```bash
mlflow ui
```
And then go to http://127.0.0.1:5000/#/ in your brower to see all the results from the experiments we runned and exact hyperparameters used in each run.

</details>


## 4. Citation
If our project is helpful for your research, please consider citing :
```

@article{Li2024sureood,
    author    = {Li, Yang and Sha, Youyang and Wu, Shengliang and Li, Yuting and Yu, Xuanlong and Huang, Shihua and Cun, Xiaodong and Chen,Yingyi and Chen, Dexiong and Shen, Xi},
    title     = {SURE-OOD: Detecting OOD samples with SURE},
    month     = {September}
    year      = {2024},
}
```


## 5. Acknowledgement
This project is heavily based on [Mammoth](https://github.com/aimagelab/mammoth) and [weight-interpolation-cl](https://github.com/jedrzejkozal/weight-interpolation-cl). We sincerely appreciate the authors of the mentioned works for sharing such great library as open-source project.
