# Module 09: Quantized Kernels (GGUF, AWQ, GPTQ & INT4/INT8)

## 1. Technical Explanation & Mathematical Intuition

Quantization reduces the precision of neural network weight values from floating point formats (FP16/BF16 - 16 bits) to low-bit integer formats (**INT8 - 8 bits**, **INT4 - 4 bits**).

### Quantization Formula (Affine Quantization)
To map a continuous floating-point range $[x_{\min}, x_{\max}]$ to an $n$-bit integer range $[0, 2^n - 1]$:
$$q = \text{round}\left(\frac{x}{S}\right) + Z$$
Where:
- $S$ (Scale Factor): $S = \frac{x_{\max} - x_{\min}}{2^n - 1}$
- $Z$ (Zero Point Offset): $Z = \text{round}\left(-\frac{x_{\min}}{S}\right)$

### Dequantization during CUDA GEMM
During matrix multiplication, low-precision weights $q$ are read from GPU VRAM at high speed (4x bandwidth reduction) and unpacked into registers:
$$\hat{x} = S \times (q - Z)$$

### Quantization Algorithms Overview
1. **GGUF (llama.cpp)**: Block-wise quantization format (Q4_K_M, Q5_K_M, Q8_0) optimized for CPU, Apple Silicon Metal, and consumer GPU execution.
2. **AWQ (Activation-aware Weight Quantization)**: Protects 1% of salient weights that correspond to large activation magnitudes, preserving accuracy on 4-bit models.
3. **GPTQ (Generalized Post-Training Quantization)**: Uses Second-Order Hessian optimization to quantize layer weights step-by-step with minimal accuracy degradation.

---

## 2. Architecture & Execution Flow

```
Model Size Comparison (70B Model):
FP16 (16-bit):  140 GB VRAM  -> Requires 2x A100 GPUs ($20,000+)
INT8 (8-bit):    70 GB VRAM  -> Fits on 1x A100 GPU
INT4 (4-bit):    35 GB VRAM  -> Fits on single RTX 3090 / RTX 4090 consumer GPU! ($1,500)

Execution Flow (AWQ / GPTQ CUDA Kernel):
Load INT4 Packed Weights (4 bits/weight) from VRAM  (4x Memory Bandwidth Reduction!)
                         |
           De-quantize INT4 -> FP16 in GPU SRAM
                         |
       Compute Matrix Multiplication via Tensor Cores
```

---

## 3. Production Trade-offs

| Quantization Format | VRAM Compression | Accuracy Loss | Primary Hardware Target |
|---|---|---|---|
| **FP16 Baseline** | 1.0x (Baseline) | 0.0% (Exact) | Datacenter GPUs |
| **INT8 (GPTQ / SmoothQuant)** | 2.0x Reduction | <0.1% Perplexity | Datacenter GPUs |
| **AWQ (4-bit)** | 3.5x Reduction | ~0.5% Perplexity | GPU Inference Servers |
| **GGUF (Q4_K_M)** | 3.8x Reduction | ~0.8% Perplexity | CPU, Metal, Edge, Local PC |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (Ollama / GGUF / vLLM / AWQ)**:
  - **Ollama** runs GGUF quantized models natively using C++ `llama.cpp` backends.
  - **vLLM** provides high-speed AWQ/GPTQ CUDA kernels using `--quantization awq`.
- **Proprietary (Google Gemini Edge & Mobile)**:
  - Google uses specialized INT4/INT8 quantization for **Gemini Nano** deployed on Android devices (Google Pixel phones).
  - Cloud Gemini endpoints use INT8 matrix engines on TPUs to maximize throughput.

---

## 5. AWS Deep Dive

- **AWS SageMaker LMI**: Supports AWQ and GPTQ quantized models out of the box. Set `option.quantization=awq` in `serving.properties`.
- **AWS EC2 Costs**: AWQ allows serving 70B models on cheaper single-GPU instances (`g5.12xlarge` or `g6.4xlarge`) instead of multi-GPU nodes, cutting server costs by **60% to 75%**.
- **AWS Neuron (Inferentia2)**: Supports INT8 and INT4 quantization via Neuron compiler flags `--quantization-type int8`.
