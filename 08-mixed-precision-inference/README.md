# Module 08: Mixed Precision Inference (FP32 vs FP16 vs BF16 vs FP8)

## 1. Technical Explanation & Mathematical Intuition

Default deep learning models were historically trained in **FP32** (Single Precision Floating Point), consuming 32 bits (4 bytes) per weight/activation parameter.

**Mixed Precision Inference** uses lower bit-width floating point formats during inference execution:
- **FP16 (Half Precision)**: 1 sign bit, 5 exponent bits, 10 mantissa bits. Consumes **2 bytes**.
- **BF16 (Bfloat16)**: 1 sign bit, 8 exponent bits (same dynamic range as FP32), 7 mantissa bits. Consumes **2 bytes**. Prevents numerical underflow/overflow without loss scaling.
- **FP8 (E4M3 / E5M2)**: 8-bit floating point format supported on Hopper (H100) and Ada Lovelace GPUs. Consumes **1 byte**.

### VRAM & Throughput Formula
Converting model weights from FP32 to FP16/BF16 instantly cuts model VRAM memory footprint by **50%**:
$$\text{Memory}_{\text{FP16}} = \frac{\text{Params} \times 2 \text{ Bytes}}{10^9} \text{ GB}$$
For example, Llama-3-70B requires:
- FP32: $70 \times 4 = 280\text{ GB VRAM}$
- FP16 / BF16: $70 \times 2 = 140\text{ GB VRAM}$
- FP8: $70 \times 1 = 70\text{ GB VRAM}$

Modern GPU **Tensor Cores** (NVIDIA Volta, Ampere, Hopper) achieve **2x to 4x higher TFLOPs** throughput when processing matrix multiplications in FP16/BF16/FP8 compared to FP32.

---

## 2. Architecture & Execution Flow

```
Weights & Input Activations (FP16 / BF16 / FP8 in GPU HBM)
                           |
            Load into Tensor Cores (SRAM)
                           |
    Compute Matrix Multiplication in FP16/FP8 Tensors (2x - 4x Faster TFLOPs!)
                           |
       Accumulate Result in FP32 / FP16 Output
```

---

## 3. Production Trade-offs

| Format | Memory per Parameter | Dynamic Range | Hardware Compatibility | Production Status |
|---|---|---|---|---|
| **FP32** | 4 Bytes | High ($10^{-38} \dots 10^{38}$) | All Hardware | Obsolete for inference |
| **FP16** | 2 Bytes | Moderate ($10^{-5} \dots 65504$) | Volta, Turing, Ampere | Legacy standard (Requires loss scaling) |
| **BF16** | 2 Bytes | High ($10^{-38} \dots 10^{38}$) | Ampere (A100), Hopper (H100) | **Modern Gold Standard** |
| **FP8** | 1 Byte | Limited (E4M3 / E5M2) | Hopper (H100), Ada (L40S) | **Emerging High-Throughput Standard** |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (PyTorch Automatic Mixed Precision `torch.cuda.amp` / vLLM)**:
  - Enabled effortlessly in PyTorch using `with torch.autocast(device_type="cuda", dtype=torch.bfloat16):`.
  - vLLM serves models in `bfloat16` by default, with `--dtype fp8` available for H100 execution.
- **Proprietary (Google Gemini Infrastructure)**:
  - Google Cloud TPUs (TPU v4/v5e/v6e) natively compute in **Bfloat16** (which Google originally invented).
  - FP8 and INT8 quantization are used in Gemini production inference microservices to serve high QPS.

---

## 5. AWS Deep Dive

- **AWS Inferentia2 (INF2)**: Supports `bfloat16`, `float16`, and dynamic range rounding modes natively inside Neuron Cores.
- **AWS SageMaker LMI Containers**: FP16 and BF16 are fully supported out-of-the-box. FP8 supported on `p5` H100 instances via `--dtype fp8`.
- **EC2 Instances**:
  - `g5` (A10G): Best for FP16 inference.
  - `g6` (L4): Best for FP8 / INT8 cost-efficient serving.
  - `p4d` (A100) & `p5` (H100): Native BF16 / FP8 acceleration.
