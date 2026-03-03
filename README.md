
<div align="center">

# IDER: Idempotent Experience Replay for Reliable Continual Learning

[![arXiv](https://img.shields.io/badge/arXiv-2603.00624-b31b1b.svg)](https://arxiv.org/abs/2603.00624)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Contact Email](https://img.shields.io/badge/Contact-Email-yellow.svg)](https://mail.google.com/mail/?view=cm&fs=1&to=zhanwnagliu@gmail.com)
<!--<a href="PROJECT_PAGE_LINK"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>-->

IDER is a novel framework for continual learning based on the idempotent property, which mitigates catastrophic forgetting and improves prediction reliability. It is a simple and robust method that can be easily integrated into other state-of-the-art approaches.

<hr/>

<p>
Zhanwang Liu<sup>1</sup><sup>*</sup>,
Yuting Li<sup>1</sup><sup>*‡</sup>,
Haoyuan Gao<sup>1</sup>,
Yexin Li<sup>4</sup>,
Linghe Kong<sup>1</sup>,
Lichao Sun<sup>3</sup>,
Weiran Huang<sup>1,2</sup><sup>†</sup>
</p>

<p>
<sup>1</sup> School of Computer Science, Shanghai Jiao Tong University<br/>
<sup>2</sup> Shanghai Innovation Institute<br/>
</p>

<p>
<sup>3</sup> Lehigh University<br/>
<sup>4</sup> State Key Laboratory of General Artificial Intelligence, BIGAI
</p>

<p>
<sup>*</sup> Equal contribution.&nbsp;&nbsp;
<sup>†</sup> Corresponding author.&nbsp;&nbsp;
<sup>‡</sup> Project lead.
</p>

<br/>

<img src="https://github.com/YutingLi0606/Idempotent-Continual-Learning/blob/main/img/method.jpg" alt="Method overview" width="600"/>

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
The example results are ResNet18 on different datasets using ER and ER+ID as baseline methods with different buffers and 0-4 seeds. All results reported here were obtained by running experiments on an NVIDIA GeForce RTX 4090.
| **Dataset** | **Buffer** | **Method** | **Forgetting(⬇️)** | **TIL(⬆️)** | **CIL(⬆️)** | **Checkpoint** |
|---|---:|---|---:|---:|---:|---|
| CIFAR-10 | 200 | ER | 59.71&nbsp;±&nbsp;2.62 | 91.48&nbsp;±&nbsp;0.93 | 48.89&nbsp;±&nbsp;2.19 | - |
|  |  | ER+ID | 16.89&nbsp;±&nbsp;2.26 | 95.87&nbsp;±&nbsp;0.36 | 70.68&nbsp;±&nbsp;1.10 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar10/buffer_200/erid_seed_0.pth) |
|  | 500 | ER | 44.75&nbsp;±&nbsp;2.94 | 93.38&nbsp;±&nbsp;0.36 | 60.62&nbsp;±&nbsp;2.46 | - |
|  |  | ER+ID | 11.59&nbsp;±&nbsp;2.13 | 96.20&nbsp;±&nbsp;0.40 | 75.52&nbsp;±&nbsp;1.35 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar10/buffer_500/erid_seed_0.pth) |
| CIFAR‑100 | 500 | ER | 73.81&nbsp;±&nbsp;0.42 | 73.98&nbsp;±&nbsp;1.15 | 21.28&nbsp;±&nbsp;1.08 | - |
|  |  | ER+ID | 32.27&nbsp;±&nbsp;1.96 | 83.30&nbsp;±&nbsp;0.41 | 45.21&nbsp;±&nbsp;1.20 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar100/buffer_500/erid_seed_0.pth) |
|  | 2000 | ER | 54.52&nbsp;±&nbsp;0.62 | 81.62&nbsp;±&nbsp;0.95 | 37.93&nbsp;±&nbsp;0.76 | - |
|  |  | ER+ID | 18.76&nbsp;±&nbsp;1.52 | 86.54&nbsp;±&nbsp;0.34 | 56.30&nbsp;±&nbsp;0.50 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar100/buffer_2000/erid_seed_0.pth) |
| Tiny‑ImageNet | 4000 | ER | 56.89&nbsp;±&nbsp;0.74 | 66.68&nbsp;±&nbsp;0.47 | 25.20&nbsp;±&nbsp;0.70 | - |
|  |  | ER+ID | 21.62&nbsp;±&nbsp;1.67 | 74.56&nbsp;±&nbsp;0.55 | 43.25&nbsp;±&nbsp;1.26 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-tinyimg/buffer_4000/erid_seed_0.pth) |

The results below were obtained using an Ascend 910B.
<details> <summary><b>ASCEND</b></summary>
    
| **Dataset** | **Buffer** | **Method** | **Forgetting(⬇️)** | **TIL(⬆️)** | **CIL(⬆️)** | **Checkpoint** |
|---|---:|---|---:|---:|---:|---|
| CIFAR-10 | 200 | ER+ID | 16.57&nbsp;±&nbsp;3.29 | 95.73&nbsp;±&nbsp;0.30 | 70.85&nbsp;±&nbsp;0.81 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar10/buffer_200/erid_seed_0.pth) |
|  | 500 | ER+ID | 12.02&nbsp;±&nbsp;1.39 | 96.07&nbsp;±&nbsp;0.19 | 75.06&nbsp;±&nbsp;0.95 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar10/buffer_500/erid_seed_0.pth) |
| CIFAR‑100 | 500 | ER+ID | 31.85&nbsp;±&nbsp;3.50 | 83.45&nbsp;±&nbsp;0.37 | 45.55&nbsp;±&nbsp;0.66 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar100/buffer_500/erid_seed_0.pth) |
|  | 2000 | ER+ID | 18.99&nbsp;±&nbsp;1.09 | 86.79&nbsp;±&nbsp;0.30 | 56.15&nbsp;±&nbsp;0.31 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-cifar100/buffer_2000/erid_seed_0.pth) |
| Tiny‑ImageNet | 4000 | ER+ID | 20.73&nbsp;±&nbsp;0.72 | 74.30&nbsp;±&nbsp;0.97 | 43.15&nbsp;±&nbsp;1.20 | [pth](https://github.com/YutingLi0606/Idempotent-Continual-Learning/tree/main/experiments/seq-tinyimg/buffer_4000/erid_seed_0.pth) |
</details>


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
@article{liu2026ider,
  title={IDER: IDempotent Experience Replay for Reliable Continual Learning},
  author={Zhanwang Liu and Yuting Li and Haoyuan Gao and Yexin Li and Linghe Kong and Lichao Sun and Weiran Huang},
  journal={arXiv preprint arXiv：2603.00624},
  year={2026}
}
```


## 5. Acknowledgement
This project is heavily based on [Mammoth](https://github.com/aimagelab/mammoth) and [weight-interpolation-cl](https://github.com/jedrzejkozal/weight-interpolation-cl). We sincerely appreciate the authors of the mentioned works for sharing such great library as open-source project.

✨ Feel free to contribute and reach out if you have any questions! ✨  
📧 Email: [zhanwnagliu@gmail.com](mailto:zhanwnagliu@gmail.com)

