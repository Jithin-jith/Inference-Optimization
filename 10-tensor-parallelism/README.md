# Module 10: Tensor Parallelism (TP)

## 1. Technical Explanation & Mathematical Intuition

When a model's parameters and KV-cache exceed the physical VRAM capacity of a single GPU (e.g. 70B FP16 model requiring 140GB VRAM), the model must be distributed across multiple GPUs.

**Tensor Parallelism (Megatron-LM style)** splits individual weight matrices (such as Linear projections in Self-Attention and MLPs) **intra-layer** across $N$ GPUs within a single node connected by high-bandwidth NVLink interconnects (900 GB/s on H100).

### Column Parallel Linear Layer (Self-Attention $W_Q, W_K, W_V$ & MLP Gate $W_1, W_3$):
Splits weight matrix $W \in \mathbb{R}^{h \times d}$ vertically into $N$ GPU shards:
$$W = \begin{bmatrix} W_1 & W_2 & \dots & W_N \end{bmatrix}$$
Each GPU $i$ computes its portion in parallel: $Y_i = X W_i$.

### Row Parallel Linear Layer (Self-Attention Output $W_O$ & MLP Down $W_2$):
Splits weight matrix horizontally across GPUs:
$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}$$
Each GPU computes $Y_i = X_i W_i$. An **All-Reduce Sum** collective operation merges outputs across GPUs:
$$Y = \text{All-Reduce-Sum}\left(\sum_{i=1}^N Y_i\right)$$

---

## 2. Architecture & Execution Flow

```
Input X (Broadcasted to GPU 0 & GPU 1)
   |
   +-----------------------+-----------------------+
   |                                               |
[GPU 0]: Y1 = X * W_col_1                      [GPU 1]: Y2 = X * W_col_2
   |                                               |
   +-----------------------+-----------------------+
                           |
[Column Parallel Output Y = [Y1, Y2]]
                           |
   +-----------------------+-----------------------+
   |                                               |
[GPU 0]: Z1 = Y1 * W_row_1                     [GPU 1]: Z2 = Y2 * W_row_2
   |                                               |
   +-----------------------+-----------------------+
                           |
            ALL-REDUCE SUM over NVLink Interconnect
                           |
             Final Output Z = Z1 + Z2
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Latency Reduction** | Slashes inference latency per token dramatically | High-frequency sub-millisecond NVLink communication overhead |
| **VRAM Distribution** | Splits weight & KV memory evenly across $N$ GPUs | **Requires NVLink / NVSwitch** (Cannot run effectively over standard PCIe or Ethernet) |
| **Serving Degree** | Typically tuned to 2, 4, or 8 GPUs per node | Limited to total GPUs within a single physical node |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (vLLM / TensorRT-LLM / DeepSpeed / Megatron)**:
  - Enabled seamlessly in vLLM using `--tensor-parallel-size 2` (or 4 / 8).
  - Uses PyTorch Distributed backend (`torch.distributed`) with NCCL over NVLink.
- **Proprietary (Google Gemini TPU Sharding)**:
  - Google JAX / XLA compiles Tensor Parallelism across TPU Pod chips (ICI - Inter-Chip Interconnect operating at 1.6 TB/s per chip).
  - Automatically shards large Gemini 2.5 Pro layers across hundreds of TPU cores transparently.

---

## 5. AWS Deep Dive

- **AWS Multi-GPU Instances**:
  - `p5.48xlarge` (8x H100 80GB GPUs with 900 GB/s NVSwitch): Supports Tensor Parallelism up to `TP=8`.
  - `p4d.24xlarge` (8x A100 40GB/80GB with NVLink): Supports `TP=2, 4, 8`.
  - `g5.12xlarge` (4x A10G 24GB): Supports `TP=2, 4` over NVLink.
- **SageMaker LMI Configuration**:
  ```properties
  option.tensor_parallel_degree=4
  option.model_id=meta-llama/Meta-Llama-3-70B-Instruct
  ```
- **AWS Neuron (Inferentia2 / Trainium1)**:
  - Neuron SDK supports Tensor Parallelism across NeuronCores via `tensor_parallel_degree=2` or `8` in `torch-neuronx`.
