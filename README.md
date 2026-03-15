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
- Conda (recommended) or compatible virtual environment manager

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
| [Kalantari et al.](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) | 74 training scenes with 3 exposure brackets | [Link](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) |
| [Sen et al.](http://web.cecs.pdx.edu/~fliu/project/robust-hdr/) | HDR dataset with large object motion | [Link](http://web.cecs.pdx.edu/~fliu/project/robust-hdr/) |

### Testing Datasets

| Dataset | Scenes | Characteristics |
|---------|--------|-----------------|
| Kalantari Test Set | 15 | Standard HDR benchmark |
| [Tursun et al.](https://userpages.cs.umbc.edu/~kayyan/papers/sig18_HDR_real_benchmark.pdf) | 79 | Large motion, saturated regions |
| [Prabhakar et al.](https://val.cds.iisc.ac.in/HDR/nightHDR/night.html) | 65 | Night-time HDR scenes |

### Directory Structure

```
datasets/
├── train/
│   ├── kalantari/
│   │   ├── input/          # LDR image sequences (3 exposures)
│   │   └── gt/             # Ground truth HDR images
│   └── sen/
│       ├── input/
│       └── gt/
└── test/
    ├── kalantari/
    ├── tursun/
    └── prabhakar/
```

---

## 🚀 Getting Started

### Training

Train the DDPF-PR model from scratch:

```bash
python train.py --config configs/ddpf_pr.yaml
```

Or use distributed training with multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train.py --config configs/ddpf_pr.yaml --launcher pytorch
```

Notes:
- Adjust `--nproc_per_node` to match the number of GPUs available.
- Change `--master_port` if the chosen port is in use.
- Edit `configs/ddpf_pr.yaml` to set dataset paths, model checkpoints, and training hyperparameters.
- Newer PyTorch versions recommend `torchrun` as an alternative to `torch.distributed.launch`.

### Testing

After training, run test with the test configuration:

```bash
python test.py --config configs/test.yaml --checkpoint checkpoints/ddpf_pr_best.pth
```

### Inference

Generate HDR results for your own LDR sequences:

```bash
python inference.py --input path/to/ldr_sequence/ --output path/to/save/hdr/ --checkpoint checkpoints/ddpf_pr_best.pth
```

---

## ✨ Qualitative Results

<details>
<summary><strong>Click to view qualitative comparison results</strong></summary>
<br>
<p align="center">
  <img src="figs/results_comparison.png" width="900" alt="Qualitative Results">
</p>
<p align="center">
  <em>Comparison with state-of-the-art methods on challenging scenes with large motion and saturation.</em>
</p>
</details>

---

## ✨ Quantitative Results

### Comparison with State-of-the-Art Methods

Results on **Kalantari Test Set** (PSNR / SSIM / HDR-VDP-2):

| Method | PSNR↑ | SSIM↑ | HDR-VDP-2↑ |
|--------|-------|-------|------------|
| Sen et al. (2012) | 40.46 | 0.9820 | 62.80 |
| Hu et al. (2013) | 41.24 | 0.9838 | 64.21 |
| Kalantari et al. (2017) | 41.84 | 0.9865 | 65.42 |
| Wu et al. (2018) | 42.11 | 0.9872 | 66.15 |
| Yan et al. (2019) | 42.45 | 0.9880 | 66.78 |
| Prabhakar et al. (2021) | 43.12 | 0.9895 | 67.85 |
| **DDPF-PR (Ours)** | **44.28** | **0.9912** | **69.34** |

### Results on Tursun Dataset (Large Motion)

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| Kalantari et al. (2017) | 40.15 | 0.9785 |
| Yan et al. (2019) | 41.23 | 0.9812 |
| **DDPF-PR (Ours)** | **42.67** | **0.9856** |

---

## 📦 Pretrained Models

Download our pretrained models:

| Model | Dataset | PSNR | Download |
|-------|---------|------|----------|
| DDPF-PR | Kalantari | 44.28 dB | [Google Drive](link) / [Baidu Pan](link) |
| DDPF-PR | Multi-dataset | 43.85 dB | [Google Drive](link) / [Baidu Pan](link) |

Place downloaded models in the `checkpoints/` directory.

---

## 📏 Troubleshooting

- **CUDA / PyTorch mismatch**: Verify installed `torch` wheel matches your CUDA toolkit version. Reinstall `torch` if necessary.
- **Distributed errors**: Ensure network ports are free and environment variables (`MASTER_ADDR`, `MASTER_PORT`) are set correctly if using multi-node setups.
- **Missing dependencies**: Inspect top-level and component `requirements` or `setup` files and install any additional packages required by specific modules.
- **Windows users**: Some shell scripts use `bash`; run them under WSL or adapt commands for PowerShell.

---

## 💖 Acknowledgment

This project is based on [HDR-GAN](https://github.com/wanghu178/HDR-GAN) and [AHDRNet](https://github.com/liuzhen03/AHDRNet). We thank the authors for their excellent works.

---

## 🤝🏼 Citation

If this code contributes to your research, please cite our work:

```bibtex
@article{zhou2025high,
  title={High Dynamic Range Imaging via Spatial-Frequency Interaction},
  author={Zhou, Weiyu and Yang, Yongqing and Hu, Tao and Hui, Pu and Jin, Jian and Cao, Yu and Yan, Qingsen and Zhang, Yanning},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TCSVT.2025.xxxxxxx}
}
```

---

## 🔆 Contact

If you have any questions, please feel free to contact:

- Qingsen Yan (Corresponding Author): [qingsenyan@nwpu.edu.cn](mailto:qingsenyan@nwpu.edu.cn)
- Weiyu Zhou: [weiyuzhou@mail.nwpu.edu.cn](mailto:weiyuzhou@mail.nwpu.edu.cn)
- Tao Hu: [taohu@mail.nwpu.edu.cn](mailto:taohu@mail.nwpu.edu.cn)

Or open an issue in this repository.
