# Module 12: Sequence Parallelism (Ring-Attention & Context Splitting)

## 1. Technical Explanation & Mathematical Intuition

For standard sequence lengths (e.g. 2,048 tokens), memory is dominated by Model Weights and KV-Caches. However, for **ultra-long contexts** (100,000 to 2,000,000+ tokens), non-Tensor-Parallel operations (such as LayerNorm, Dropout, and intermediate activations) consume enormous VRAM when replicated across all GPUs.

**Sequence Parallelism (Megatron-SP & Ring-Attention)** splits the sequence dimension $S$ across $N$ GPUs:
$$\text{Sequence Tokens per GPU} = \frac{S}{N}$$

### Ring Self-Attention Mechanism
1. Each GPU $i$ holds a slice of input sequence tokens $S_i = \left[ \frac{i \cdot S}{N} \dots \frac{(i+1) \cdot S}{N} \right]$.
2. Computes local Query ($Q_i$), Key ($K_i$), and Value ($V_i$) blocks.
3. Key ($K$) and Value ($V$) blocks are passed around GPUs in a **ring topology** via asynchronous P2P network transfers while local attention blocks are computed simultaneously.

---

## 2. Architecture & Execution Flow

```
Sequence Splitting across 4 GPUs (Sequence Length = 1,000,000 tokens):
GPU 0: Tokens [0 ..... 250k]   (Computes local Q0, K0, V0)
GPU 1: Tokens [250k .. 500k]   (Computes local Q1, K1, V1)
GPU 2: Tokens [500k .. 750k]   (Computes local Q2, K2, V2)
GPU 3: Tokens [750k .. 1000k]  (Computes local Q3, K3, V3)

Ring Transfer Cycle:
GPU 0 sends K0,V0 to GPU 1  --->  GPU 1 computes Attn(Q1, K0, V0)
GPU 1 sends K1,V1 to GPU 2  --->  GPU 2 computes Attn(Q2, K1, V1) ...
Overlaps Ring Communication with SRAM Computation seamlessly!
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Ultra-Long Contexts** | Unlocks 1M+ to 10M+ token processing on open weights | Complex ring network communication logic |
| **VRAM Linear Scaling** | Prevents $O(N)$ sequence activation memory spikes per GPU | Requires fast inter-GPU interconnects (NVLink / EFA) |
| **Prefill Speedup** | Distributes prompt prefill workload evenly | Overlaps communication with compute (requires precise kernel tuning) |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (DeepSpeed Sequence Parallelism / Megatron / LightSeq / vLLM)**:
  - Frameworks use Ring-Attention or Megatron-SP to split prefill context across GPU clusters.
- **Proprietary (Google Gemini 1M-2M Context Engine)**:
  - Gemini's breakthrough 1,000,000+ token context capability is built on TPU Ring-Attention sequence splitting across TPU Pod topologies.

---

## 5. AWS Deep Dive

- **EC2 `p5.48xlarge` (8x H100)**: NVSwitch ring communication topology enables near 100% linear throughput scaling for sequence-parallel long prefill tasks.
- **AWS SageMaker LMI**: Supports sequence parallelism flags in DeepSpeed and vLLM engines for long document processing.
