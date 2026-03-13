# pairgen ✂️

[![PyPI Version](https://img.shields.io/pypi/v/pairgen.svg)](https://pypi.org/project/pairgen/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pairgen.svg)](https://pypi.org/project/pairgen/)
[![License](https://img.shields.io/pypi/l/pairgen.svg)](https://pypi.org/project/pairgen/)
[![CI Status](https://github.com/ash1ra/pairgen/actions/workflows/ci.yml/badge.svg)](https://github.com/ash1ra/pairgen/actions)

**pairgen** is a fast, reliable, and dependency-light CLI tool designed to generate High-Resolution (HR) and Low-Resolution (LR) image pairs for Super-Resolution Machine Learning and Deep Learning datasets.

Whether you are preparing standard academic benchmarks (like Set5, Set14, DIV2K) or generating complex Real-World SR validation sets, `pairgen` handles exact interpolation, patch extraction, and advanced degradations out of the box.

## ✨ Features

* 🔬 **Academic Reproducibility**: Includes a pure NumPy implementation of MATLAB's `imresize` (bicubic interpolation with antialiasing), which is strictly required for evaluating standard SR models.
* 🌪️ **BSRGAN-like Degradation Pipeline**: Simulates real-world image degradation by applying random Gaussian/Sinc blur, Gaussian/Poisson noise, and JPEG compression in a **randomized order** (Random Shuffle Strategy).
* ✂️ **Smart Patch Extraction**: Easily extract multiple random crops from large images to maximize your dataset's utility.
* ⚡ **Multiprocessing**: Built with `ProcessPoolExecutor` to utilize all CPU cores, effortlessly processing thousands of high-resolution images.
* 🪶 **Lightweight**: No heavy dependencies like OpenCV or SciPy. All complex FFT math and kernels are implemented using `numpy` and `Pillow`.

## 🎯 Motivation

While working on Image Super-Resolution projects, preparing testing and validation datasets is always a bottleneck. Academic benchmarks require exact MATLAB-like downsampling, while Real-World SR evaluation requires complex degradation pipelines.

Applying degradations on-the-fly during training is a standard practice, but for **Validation and Test sets**, you need strictly fixed, pre-generated LR images to reliably compare models and calculate PSNR/SSIM metrics. `pairgen` was created to standardize this offline generation process into one CLI command, perfectly complementing tools like [manigen](https://github.com/ash1ra/manigen).

## 📦 Installation

You can install `pairgen` directly from PyPI using `pip`:

```bash
pip install pairgen
```

Or, if you use uv (recommended for CLI tools):

```bash
uv tool install pairgen
```

## 🚀 Quick Start

Generate a standard benchmark dataset with MATLAB bicubic downsampling (x4 scale):

```bash
pairgen -i data/Set14 -o data/Set14_pairs -s 4
```

## 💡 Advanced Usage Examples

#### 1. Generating Real-World SR Validation Sets

Create a complex, degraded test set by enabling the BSRGAN-like pipeline (blur, noise, and JPEG compression applied in random order):

```bash
pairgen -i data/validation -o data/validation_degraded -s 4 --blur --noise --jpeg
```

#### 2. Extracting Multiple Patches

If you have 2K/4K images and want to extract 50 random 256x256 patches from each image to build a rich dataset:

```bash
pairgen -i data/DIV2K_train -o data/DIV2K_patches -s 4 -p 256 -np 50
```

#### 3. Combining with `manigen`

`pairgen` natively supports reading file manifests (`.txt` files containing lists of paths). You can use [manigen](https://github.com/ash1ra/manigen) to index your dataset and split it, then pass the manifest to `pairgen`:

```bash
pairgen -i train_manifest.txt -o data/train_set -s 4
```

## 🛠️ CLI Reference

| Argument | Short | Description | Default
| --- | :---: | --- | :---:
| `--input-path` | `-i` | **(Required)** Input directory or manifest file to scan. | -
| `--output-dir` | `-o` | **(Required)** Output directory where HR and LR folders will be created. | -
| `--scaling-factor` | `-s` | **(Required)** Scaling factor for LR images (e.g., 2, 4). | -
| `--recursive` | `-r` | Scan subdirectories recursively. | `False`
| `--workers` | `-w` | Number of CPU cores to use. Use 1 for strict sequential order. | `1`
| `--interpolation` | `-im` | Interpolation: `matlab_bicubic`, `bilinear`, `bicubic`, `lanczos`, `nearest`. | `matlab_bicubic`
| `--patch-size` | `-p` | If > 0, extracts square patches of this size from HR. | `0`
| `--num-patches` | `-np` | Number of random patches to extract per image. | `1`
| `--augment` |   | Apply random flips and rotations to HR. | `False`
| `--blur` |   | Apply random Gaussian/Sinc blur to LR. | `False`
| `--noise` |   | Apply random Gaussian/Poisson noise to LR. | `False`
| `--jpeg` |   | Apply random JPEG compression to LR. | `False`

## 🤝 Contributing

### 1. Clone the repository

```bash
git clone https://github.com/ash1ra/pairgen
cd pairgen
```

### 2. Install dependencies using uv

```bash
uv sync
# On Windows
.venv\Scripts\activate
# on Unix or MacOS
source .venv/bin/activate
```

### 3. Format and lint the code 

```bash
uv run ruff format .
uv run ruff check .
```

### 4. Run the tests 

```bash
uv run pytest tests/ -v
```

### 5. Submit a pull request

If you'd like to contribute, please fork the repository and open a pull request to the `main` branch.
