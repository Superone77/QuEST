# QuEST: Stable Training of LLMs with 1-Bit Weights and Activations

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2502.05003)

## Quickstart 

Create a conda environment and install dependencies (we recommend Python 3.10):

```bash
conda create -n env python=3.10
conda activate env
pip install -r requirements.txt
```

Run a simple training on the SlimPajama 6B dataset:
```bash
python ./src/main.py
```

The above command trains a 213.34M parameters model with the Llama-style architecture. We recommend to use the `--compile` flag that speeds up training noticeably (up to 20% in our setup).

## Quantization

See `train.sh` for an example on how to run quantized training.

## INT4 Inference Kernels

We provide Triton/CUDA kernels for INT4 Inference on NVIDIA Ampere GPUs. The code is particularly optimized for RTX 4090.

**[Work in Progress]** The floating point numbers are currently hardcoded to type float16 in the matrix multiplication and dequantization CUTLASS template. We are adapting it to type bfloat16. However, the difference in the data types should not affect the speedup benchmarks.

### Install

#### Dependencies

- cmake
- C++ compiler (GCC/clang/...)
- nvcc

#### Instructions

```bash
cd gemm-quest
pip install -e .
```

### Benchmark

After installation, please copy the Python scripts in the `gemm-quest/quest` folder to the `src` folder in our main QuEST project folder.

You can then run the layerwise and end-to-end benchmarks using these scripts. You may need to adjust some configurations in the files to adapt your machine.

## Source Code

This repository is based on the [epfml/schedules-and-scaling](https://github.com/epfml/schedules-and-scaling) repository for their "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" paper. We thank the original creators for making public and open-licenced. 

