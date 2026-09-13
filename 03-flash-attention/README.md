# Module 03: FlashAttention (v1 / v2 / v3)

## 1. Technical Explanation & Mathematical Intuition

Standard Softmax Attention computes:
$$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right), \quad O = A V$$

In standard PyTorch, this materializes intermediate tensors $S = Q K^T \in \mathbb{R}^{N \times N}$ and $A = \text{softmax}(S) \in \mathbb{R}^{N \times N}$ in High Bandwidth Memory (HBM).
For sequence length $N$, memory reads/writes to HBM scale as $O(N^2)$. Since HBM read/write speed (~2 TB/s on A100) is far slower than GPU SRAM compute (~19.5 TB/s), standard attention is severely **memory-bandwidth bound**.

### FlashAttention Solution (Tiling & Online Softmax):
FlashAttention divides $Q, K, V$ matrices into smaller blocks (tiles) that fit inside GPU **SRAM** (19 MB per SM on A100):
1. Loads a block of $Q$ and blocks of $K, V$ into SRAM.
2. Computes partial attention scores locally inside SRAM using **Online Softmax Scaling**:
   $$m_{\text{new}} = \max(m_{\text{prev}}, \tilde{m}), \quad d_{\text{new}} = d_{\text{prev}} e^{m_{\text{prev}} - m_{\text{new}}} + \tilde{d} e^{\tilde{m} - m_{\text{new}}}$$
3. Accumulates final output $O$ without ever saving the full $N \times N$ attention matrix $A$ back to HBM.

### Complexity Comparison
- **Standard Attention HBM Memory Access**: $O(N^2 d)$
- **FlashAttention HBM Memory Access**: $O(N^2 d^2 M^{-1})$ (where $M$ is SRAM size)
- **Memory Footprint**: Reduced from $O(N^2)$ to $O(N)$ (linear with sequence length).

---

## 2. Architecture & Execution Flow

```
Standard Attention (HBM Memory Bottleneck):
Q, K (HBM) ---> GPU SRAM ---> Write NxN Matrix S back to HBM (SLOW!)
S (HBM)     ---> GPU SRAM ---> Write Softmax Matrix A back to HBM (SLOW!)
A, V (HBM)  ---> GPU SRAM ---> Write Output O to HBM

FlashAttention (SRAM Tiling - Zero NxN HBM Writes):
Q_block, K_block, V_block (HBM) ---> Load into Fast SRAM Tile
                                       |
                     Compute Local Online Softmax in SRAM
                                       |
                               Accumulate into O_block
                                       |
Only Final Output O_block is written back to HBM! (Speedup: 2x - 4x)
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Speed** | 2x to 4x faster prefill execution | Requires Ampere (A100/A10G) or Hopper (H100) architecture |
| **VRAM Usage** | Slashes VRAM from $O(N^2)$ to $O(N)$ | Requires head dimensions to be powers of 2 (e.g. 64, 128) |
| **Precision** | 100% exact numerical match (no approximation) | FlashAttention-3 requires CUDA 12+ and Hopper GPUs |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (PyTorch / vLLM / Ollama)**:
  - Integrated via PyTorch `torch.nn.functional.scaled_dot_product_attention` (SDPA) which automatically enables FlashAttention kernels when GPU supports it.
  - vLLM natively uses FlashAttention-2 / FlashDecoding for ultra-fast prefill and long-context decoding.
- **Proprietary (Google Gemini API)**:
  - Google's TPU architectures (TPU v4, TPU v5e, TPU v6e) use **Ring-Attention** and **XLA Attention Fusions** compiled via JAX/XLA, achieving similar tile-based $O(N)$ HBM memory reduction to serve Gemini's 1M-2M token context windows.

---

## 5. AWS Deep Dive

- **EC2 Instance Types**:
  - `p4d.24xlarge` (A100 40GB/80GB): Supports FlashAttention-1 & FlashAttention-2.
  - `p5.48xlarge` (H100 80GB): Supports FlashAttention-3 with FP8 Tensor Cores.
  - `g5.12xlarge` (A10G): Supports FlashAttention-2 via PyTorch SDPA.
- **AWS Neuron (Inferentia2 / Trainium1)**:
  - Uses Neuron Compiler (`neuronx-cc`) which automatically performs attention fusion tiles into Neuron Core Scratchpad RAM.
