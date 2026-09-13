# Module 01: Key-Value (KV) Caching

## 1. Technical Explanation & Mathematical Intuition

During autoregressive language generation, the model predicts one token at a time. To generate token $t_i$, the attention mechanism computes attention scores between $t_i$ and all previous tokens $t_1, \dots, t_{i-1}$.

Without KV-Caching, at step $i$, the projections for Key ($K$) and Value ($V$) tensors for all preceding tokens $1 \dots i-1$ are recomputed from scratch.
For a sequence of length $N$, the computational complexity of attention across all steps becomes:
$$\text{Compute Complexity without KV-Cache} = O(N^3)$$

By caching the Key ($K$) and Value ($V$) state vectors of past tokens in GPU VRAM, only the Query ($Q$), Key ($K_i$), and Value ($V_i$) for the *new single token* $t_i$ need to be computed at step $i$.
$$\text{Compute Complexity with KV-Cache} = O(N^2)$$

### Memory Footprint Formula
The memory required to store the KV-Cache for a batch of requests is:
$$\text{Memory}_{\text{KV}} = 2 \times b \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times s \times p$$
Where:
- $b$: Batch size
- $n_{\text{layers}}$: Number of transformer layers
- $n_{\text{heads}}$: Number of attention heads (or KV heads in MQA/GQA)
- $d_{\text{head}}$: Head dimension ($\text{hidden\_size} / n_{\text{heads}}$)
- $s$: Sequence length
- $p$: Precision in bytes (2 bytes for FP16/BF16)

---

## 2. Architecture & Execution Flow

```
Without KV-Cache (Recompute every step):
Step 1: [T1] -> Compute K1, V1
Step 2: [T1, T2] -> Recompute K1, V1 + Compute K2, V2  (Wasted FLOPs)
Step 3: [T1, T2, T3] -> Recompute K1, V1, K2, V2 + Compute K3, V3

With KV-Cache:
Step 1: [T1] -> Compute & Store (K1, V1) in Cache RAM
Step 2: [T2] -> Fetch (K1, V1) from Cache + Compute & Append (K2, V2)
Step 3: [T3] -> Fetch (K1..K2, V1..V2) + Compute & Append (K3, V3)
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Inter-Token Latency (ITL)** | Massive reduction (10x-50x faster generation) | High VRAM consumption |
| **Throughput** | High per-GPU efficiency | Potential out-of-memory (OOM) at high batch sizes |
| **Context Length** | Enables multi-thousand token contexts | Linear growth in memory footprint per sequence |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (Ollama / PyTorch / vLLM)**:
  Ollama and llama.cpp manage KV-cache allocation automatically in GPU VRAM. Modern open models also use **Grouped-Query Attention (GQA)** or **Multi-Query Attention (MQA)** to reduce the KV-cache size by sharing KV heads across multiple query heads.
- **Proprietary (Google Gemini API)**:
  Cloud APIs abstract GPU VRAM. Gemini provides **Explicit Context Caching**, allowing developers to pre-load large system prompts, PDF documents, or codebases into Gemini's KV cache servers, reducing cost by up to 75% and cutting TTFT.

---

## 5. AWS Deep Dive

- **AWS Inferentia2 / Trainium1**: AWS Neuron SDK requires explicit memory allocation for KV cache tensor shapes (`seq_len`). Neuron Core utilizes static KV cache buffers allocated at engine load time.
- **AWS Bedrock Prompt Caching**: Automatically detects matching prefix prompts (>1024 tokens) and reuses cached KV states across requests, lowering latency and cost.
- **SageMaker LMI**: Uses vLLM or TensorRT-LLM backends where KV-cache allocation can be fine-tuned via `gpu_memory_utilization` and `max_model_len` settings.
