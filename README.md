# High Dynamic Range Imaging via Spatial-Frequency Interaction

[![IEEE TCSVT](https://img.shields.io/badge/IEEE-TCSVT-blue.svg)](https://ieeexplore.ieee.org/document/11390683)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> [Weiyu Zhou](https://scholar.google.com/citations?user=UAAtWOIAAAAJ)<sup>1</sup>, [Yongqing Yang](https://scholar.google.com/citations?user=xxx)<sup>1</sup>, [Tao Hu](https://scholar.google.com/citations?user=BNkFUbsAAAAJ)<sup>1</sup>, Pu Hui<sup>1</sup>, [Jian Jin](https://jianjin008.github.io/)<sup>2</sup>, Yu Cao<sup>1</sup>, [Qingsen Yan](https://scholar.google.com/citations?hl=zh-CN&user=BSGy3foAAAAJ)<sup>12*</sup>, Yanning Zhang<sup>1</sup><br>
> <sup>1</sup>Northwestern Polytechnical University &nbsp; <sup>2</sup>Shenzhen Research Institute of Northwestern Polytechnical University &nbsp; <sup>*</sup>Corresponding Author

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11390683">📜IEEE Xplore</a> |
  <a href="#">🌐Project Page (Coming Soon)</a> |
  <a href="#">📹Demo Video</a>
</p>

---

## 📋 Abstract

High Dynamic Range (HDR) imaging aims to reconstruct high-quality HDR images from multiple Low Dynamic Range (LDR) images with different exposures. Existing methods primarily focus on spatial domain alignment and fusion, often neglecting the critical frequency information that is essential for preserving fine details and textures. To address these challenges, we introduce a **Dual-Domain Parallel Fusion Network with Progressive Refinement (DDPF-PR)** that explicitly models spatial-frequency interaction for HDR reconstruction. Our framework decouples feature learning into spatial and frequency branches, enabling complementary information aggregation from both domains. Extensive experiments demonstrate that DDPF-PR consistently outperforms state-of-the-art methods on standard HDR benchmarks.

---

## 🔥 Update Log

- **[2025/02]** 📢 📢 Paper accepted by IEEE TCSVT!
- **[2025/03]** 🚀 Code and pretrained models released.

---

## 📖 Method Overview

The proposed **DDPF-PR** framework consists of three key components:
- **Spatial Branch**: Captures local spatial details and structural information
- **Frequency Branch**: Models global frequency characteristics and texture patterns
- **Progressive Refinement Module**: Fuses dual-domain features with iterative enhancement

<p align="center">
  <img src="figs/overview.png" width="900" alt="DDPF-PR Architecture">
</p>

---

## 🛠️ Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU training)
- [Accelerate](https://huggingface.co/docs/accelerate/index) (for distributed training)

---

## 🌍 Create Environment

Create and activate the Conda environment named `ddpf` with Python 3.8:

```bash
conda create -n ddpf python=3.8 -y
conda activate ddpf
```

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## ⬇️ Dataset Preparation

Prepare the train and test datasets following the **Datasets** section in our paper.

We evaluate our method on the following HDR datasets:

### Training Datasets

| Dataset | Description | Download |
|---------|-------------|----------|
| [Kalantari et al. (SIG17)](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) | 74 training scenes with 3 exposure brackets | [Link](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) |
| Hu et al. | Real-world challenging scenes | - |
| TEL | Extreme dynamic range scenes | - |

### Testing Datasets

| Dataset | Scenes | Characteristics |
|---------|--------|-----------------|
| Kalantari Test Set | 15 | Standard HDR benchmark |
| Tursun et al. | 79 | Large motion, saturated regions |
| Prabhakar et al. | 65 | Night-time HDR scenes |
| Hu et al. | 20 | Real-world challenging scenes |
| TEL | 10 | Extreme dynamic range scenes |

### Directory Structure

```
datasets/
├── train/
│   ├── kalantari/
│   │   ├── training/          # LDR image sequences (3 exposures)
│   │   └── val/               # Ground truth HDR images
│   ├── Hu/
│   │   ├── training/
│   │   └── val/
│   └── TEL/
│       ├── training/
│       └── val/
└── test/
    ├── kalantari/             # 15 test scenes
    ├── Hu/                    # 20 real-world challenging scenes
    ├── TEL/                   # 10 extreme dynamic range scenes
    ├── Tursun/                # 79 large motion scenes
    └── Prabhakar/             # 65 night-time scenes
```

---

## 🚀 Getting Started

### Training

Train the DDPF-PR model using Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py --dataset_dir /path/to/dataset --logdir /path/to/log
```

For single GPU training:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py --dataset_dir /path/to/dataset --logdir /path/to/log
```

Key training arguments:
- `--dataset_dir`: Path to the training dataset
- `--logdir`: Directory to save checkpoints and logs
- `--batch_size`: Training batch size (default: 12)
- `--lr`: Learning rate (default: 0.0002)
- `--epochs`: Number of training epochs (default: 100)
- `--resume`: Path to checkpoint for resuming training

### Testing

Run inference on test datasets:

```bash
python fullimagetest.py --checkpoint /path/to/checkpoint.pth --input /path/to/test/data --output /path/to/results
```

---

## ✨ Qualitative Results

<details>
<summary><strong>Kalantari Dataset Results</strong></summary>
<br>
<p align="center">
  <img src="figs/sig.png" width="900" alt="Results on Kalantari Dataset">
</p>
<p align="center">
  <em>Visual comparison on Kalantari dataset with large motion and saturation.</em>
</p>
</details>

<details>
<summary><strong>Tursun Dataset Results</strong></summary>
<br>
<p align="center">
  <img src="figs/sen_turn.png" width="900" alt="Results on Tursun Dataset">
</p>
<p align="center">
  <em>Visual comparison on Tursun dataset with large motion.</em>
</p>
</details>

<details>
<summary><strong>Hu and TEL Dataset Results</strong></summary>
<br>
<p align="center">
  <img src="figs/tel_hu.png" width="900" alt="Results on Hu and TEL Datasets">
</p>
<p align="center">
  <em>Visual comparison on Hu and TEL datasets with extreme dynamic range.</em>
</p>
</details>

---

## ✨ Quantitative Results

### Results on Kalantari Test Set

| Method | PSNR-μ↑ | PSNR-l↑ | SSIM-μ↑ | SSIM-l↑ | HDR-VDP-2↑ |
|--------|---------|---------|---------|---------|------------|
| DHDR [41] | 41.64 | 40.91 | 0.9869 | 0.9858 | 60.30 |
| AHDR [20] | 43.62 | 41.03 | 0.9900 | 0.9862 | 62.30 |
| NHDRR [42] | 42.41 | 41.08 | 0.9887 | 0.9861 | 61.21 |
| HDR-GAN [34] | 43.92 | 41.57 | 0.9905 | 0.9865 | 65.45 |
| APNT [30] | 43.94 | 41.61 | 0.9898 | 0.9879 | 64.05 |
| CA-ViT [18] | 44.32 | 42.18 | 0.9916 | 0.9884 | 66.03 |
| HyHDR [43] | 44.64 | 42.47 | 0.9915 | 0.9894 | 66.05 |
| SCTNet [2] | 44.43 | 42.21 | 0.9918 | 0.9891 | 66.64 |
| DiffHDR [36] | 44.11 | 41.73 | 0.9911 | 0.9885 | 65.52 |
| SAFNet [44] | 44.66 | 43.18 | 0.9919 | 0.9901 | 66.11 |
| LFDiff [21] | 44.76 | 42.59 | 0.9919 | 0.9906 | 66.54 |
| AFUNet [33] | 44.91 | 42.59 | 0.9923 | 0.9906 | 66.75 |
| **DDPF-PR (Ours)** | **44.93** | **42.61** | **0.9922** | **0.9908** | **66.78** |

### Results on TEL Dataset

| Method | PSNR-μ↑ | PSNR-l↑ | SSIM-μ↑ | SSIM-l↑ | HDR-VDP-2↑ |
|--------|---------|---------|---------|---------|------------|
| DHDR [41] | 40.05 | 43.37 | 0.9794 | 0.9924 | 67.09 |
| AHDR [20] | 42.08 | 45.30 | 0.9837 | 0.9943 | 68.80 |
| NHDRR [42] | 36.68 | 39.61 | 0.9590 | 0.9853 | 65.41 |
| HDR-GAN [34] | 41.71 | 44.87 | 0.9832 | 0.9949 | 69.57 |
| CA-ViT [18] | 42.39 | 46.35 | 0.9848 | 0.9948 | 69.23 |
| SCTNet [2] | 42.55 | 47.51 | 0.9850 | 0.9952 | 70.66 |
| DiffHDR [36] | 42.18 | 45.63 | 0.9841 | 0.9946 | 69.88 |
| SAFNet [44] | 42.68 | 47.46 | 0.9792 | 0.9955 | 68.16 |
| AFUNet [33] | 43.31 | 47.83 | 0.9876 | 0.9959 | 71.08 |
| **DDPF-PR (Ours)** | **43.49** | **48.25** | **0.9878** | **0.9961** | **70.96** |

### Results on Hu Dataset

| Method | PSNR-μ↑ | PSNR-l↑ | SSIM-μ↑ | SSIM-l↑ | HDR-VDP-2↑ |
|--------|---------|---------|---------|---------|------------|
| DHDR [41] | 41.13 | 41.20 | 0.9870 | 0.9941 | 70.82 |
| AHDR [20] | 45.76 | 49.22 | 0.9956 | 0.9980 | 75.04 |
| NHDRR [42] | 45.15 | 48.75 | 0.9956 | 0.9981 | 74.86 |
| HDR-GAN [34] | 45.86 | 49.14 | 0.9945 | 0.9989 | 75.19 |
| APNT [30] | 46.41 | 47.97 | 0.9953 | 0.9986 | 73.06 |
| CA-ViT [18] | 48.10 | 51.17 | 0.9947 | 0.9989 | 77.12 |
| HyHDR [43] | 48.46 | 51.91 | 0.9959 | 0.9991 | 77.24 |
| DiffHDR [36] | 48.03 | 50.23 | 0.9954 | 0.9989 | 76.22 |
| SCTNet [2] | 48.10 | 51.03 | 0.9963 | 0.9991 | 77.14 |
| SAFNet [44] | 47.18 | 49.35 | 0.9951 | 0.9990 | 76.83 |
| LFDiff [21] | 48.74 | 52.10 | 0.9968 | 0.9993 | 77.35 |
| AFUNet [33] | 48.83 | 52.13 | 0.9968 | 0.9991 | 77.44 |
| **DDPF-PR (Ours)** | **48.86** | **52.42** | **0.9969** | **0.9992** | **77.46** |

---

## 📦 Pretrained Models

Download our pretrained model:

**Baidu Pan**: [44.95sig17.pth](https://pan.baidu.com/s/1z5-vvnn6H4Ep_qWuZILn-w) (提取码: zxat)

Place the downloaded checkpoint in your working directory and specify the path when running `fullimagetest.py`.

---

## 📏 Troubleshooting

- **CUDA / PyTorch mismatch**: Verify installed `torch` wheel matches your CUDA toolkit version. Reinstall `torch` if necessary.
- **Accelerate configuration**: Run `accelerate config` to set up distributed training configuration.
- **Out of memory**: Reduce `--batch_size` or use gradient accumulation.
- **Windows users**: Some shell scripts use `bash`; run them under WSL or adapt commands for PowerShell.

---

## 💖 Acknowledgment

We thank Qingsen Yan and Tao Hu for their support and guidance throughout this research.

---

## 🤝🏼 Citation

If this code contributes to your research, please cite our work:

```bibtex
@article{zhou2026high,
  title={High Dynamic Range Imaging via Spatial-Frequency Interaction},
  author={Zhou, Weiyu and Yang, Yongqing and Hu, Tao and Hui, Pu and Jin, Jian and Cao, Yu and Yan, Qingsen and Zhang, Yanning},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

---

## 🔆 Contact

If you have any questions, please feel free to contact:

- Weiyu Zhou: [weiyuzhou@mail.nwpu.edu.cn](mailto:weiyuzhou@mail.nwpu.edu.cn)

Or open an issue in this repository.
