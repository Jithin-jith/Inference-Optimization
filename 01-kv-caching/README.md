# Module 01: Key-Value (KV) Caching

## 1. Technical Explanation & Mathematical Intuition

During autoregressive language generation, the model predicts one token at a time. To generate token $t_i$, the self-attention mechanism calculates attention weights between $t_i$ and all previous tokens $t_1, \dots, t_{i-1}$.

### The KV Bottleneck Without Caching
Without KV-Caching, at decoding step $i$, the linear projections for Key ($K$) and Value ($V$) tensors for all historical tokens $1 \dots i-1$ are recalculated from scratch.

For a sequence of length $N$, the cumulative computational complexity across all generation steps scales cubically:
$$\text{Compute Complexity without KV-Cache} = O(N^3)$$

### The Solution With KV-Caching
By retaining the computed Key ($K$) and Value ($V$) tensors in GPU VRAM (or server-side memory), step $i$ only computes the Query ($Q_i$), Key ($K_i$), and Value ($V_i$) for the **new incoming token** $t_i$. The new $K_i$ and $V_i$ are concatenated to the historical KV cache matrix.

This reduces the overall generation compute complexity to quadratic scaling:
$$\text{Compute Complexity with KV-Cache} = O(N^2)$$

### Memory Footprint Formula
While KV caching slashes FLOPs, it trades compute for GPU VRAM capacity. The memory required to store the KV-Cache across a batch of requests is calculated as:

$$\text{Memory}_{\text{KV}} = 2 \times b \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times s \times p$$

Where:
- $b$: Batch size (number of active requests)
- $n_{\text{layers}}$: Number of transformer layers
- $n_{\text{heads}}$: Number of attention heads per layer (or KV heads in GQA/MQA)
- $d_{\text{head}}$: Head dimension ($\text{hidden\_size} / n_{\text{heads}}$)
- $s$: Sequence length (prompt + generated tokens)
- $p$: Precision in bytes (e.g., 2 bytes for FP16/BF16, 1 byte for FP8, 0.5 bytes for INT4)

---

## 2. Architecture & Execution Flow

```
WITHOUT KV-Cache (Recompute full sequence at every step):
Step 1: [T1]          -> Compute K1, V1
Step 2: [T1, T2]      -> Recompute K1, V1 + Compute K2, V2  (Wasted FLOPs)
Step 3: [T1, T2, T3]  -> Recompute K1, V1, K2, V2 + Compute K3, V3

WITH KV-Cache (Store past K, V; project ONLY 1 new token):
Step 1: [T1]          -> Compute & Store (K1, V1) in Cache RAM
Step 2: [T2]          -> Fetch (K1, V1) from Cache + Compute & Append (K2, V2)
Step 3: [T3]          -> Fetch (K1..K2, V1..V2) + Compute & Append (K3, V3)
```

---

## 3. Comprehensive Comparison: Cached vs. Uncached Execution

| Parameter | Uncached Execution | KV-Cached Execution |
|---|---|---|
| **Step Computational Complexity** | $O(i \cdot d^2)$ per step (Re-evaluates full sequence) | $O(d^2)$ per step (Evaluates only 1 token) |
| **Total Generation Complexity** | $O(N^3)$ total FLOPs | $O(N^2)$ total FLOPs |
| **Inter-Token Latency (ITL)** | High & grows linearly with sequence length | Low & stays virtually constant across tokens |
| **Time to First Token (TTFT)** | Baseline prompt prefill latency | Slashed by 50%–80% when reusing pre-cached prompt prefix |
| **API Token Cost (Cloud APIs)** | Charged 100% price for prompt tokens every request | **75%–80% discount** on pre-cached prompt tokens |
| **VRAM Memory Requirement** | Low (only activation memory during forward pass) | **High** (grows dynamically with sequence length & batch size) |
| **Out-Of-Memory (OOM) Risk** | Low risk during decode phase | **High risk** under large batch sizes or long contexts |

---

## 4. When to Use vs. When NOT to Use KV Caching

### When to Use KV Caching (Ideal Scenarios)

1. **Multi-Turn Chat Conversations**:
   - Chatbots retaining conversation history across multiple turns. Reusing pre-computed KV states avoids re-processing earlier turns.
2. **Document Q&A & RAG Pipelines**:
   - Querying large static documents (e.g., 50k–100k token PDFs, books, financial reports, or API docs) with multiple sequential user prompts.
3. **Heavy System Instructions & Agent Schemas**:
   - Enterprise AI agents with massive system prompts, function/tool-calling schemas, or blueprint templates.
4. **Codebase Analysis**:
   - Developer tools maintaining an entire repository in context while asking multiple follow-up refactoring or debugging questions.
5. **High-Throughput Autoregressive Token Generation**:
   - Essential for any production LLM service serving standard text completion to keep Inter-Token Latency (ITL) low.

### When NOT to Use KV Caching (Anti-Patterns & Edge Cases)

1. **Single-Shot Short Queries Without Shared Context**:
   - Simple standalone tasks like translating a 5-word sentence or classifying single short text snippets with no system prompt reuse. The overhead of cache management exceeds any benefits.
2. **Highly Dynamic, Non-Repeating Prompts**:
   - If every API request contains completely distinct, non-overlapping long prompts, pre-caching will not hit any cache matches.
3. **Severe VRAM Constraints Without PagedAttention**:
   - In self-hosted inference servers running at high concurrency, allocating full contiguous KV cache buffers for long sequences can trigger GPU Out-Of-Memory (OOM) errors, severely limiting maximum batch size.
4. **Non-Autoregressive / Single Forward Pass Tasks**:
   - Embedding models (e.g., BERT, bge-large) or single forward pass classifiers do not perform step-by-step token generation, making KV caching irrelevant.
5. **Infrequently Queried Large Contexts on Paid Cloud APIs**:
   - In cloud APIs like Google Gemini, creating an explicit cache resource incurs storage/TTL costs. If a context is only queried once before expiring, creating a cache resource is inefficient.

---

## 5. Open-Source vs. Proprietary Paradigm

- **Open-Source (PyTorch / Ollama / vLLM)**:
  - Frameworks manage KV-cache allocation directly in GPU VRAM.
  - Modern open-source models use **Grouped-Query Attention (GQA)** (e.g., Llama 3) or **Multi-Query Attention (MQA)** to reduce the number of KV heads by $4\times$ to $8\times$, shrinking the KV cache memory footprint proportionally.
  - Engines like **vLLM** introduce **PagedAttention** to eliminate VRAM fragmentation by chunking the KV cache into virtual memory pages.

- **Proprietary (Google Gemini API)**:
  - Cloud APIs abstract hardware VRAM completely.
  - Google Gemini provides **Explicit Context Caching** (`client.caches.create()`), persisting KV states on Google TPU clusters for a designated Time-to-Live (TTL). This cuts input token costs by up to 80% and slashes Time-to-First-Token (TTFT).

---

## 6. Repository Code Implementation Walkthrough

This module contains two executable reference benchmarks:

### 1. Open-Source Implementation: [opensource_ollama.py](Inference-Optimization/01-kv-caching/opensource_ollama.py)
- **`torch_kv_cache_demo()`**: Pure PyTorch benchmark building multi-head self-attention matrices manually. Compares $O(N^3)$ sequence re-computation against $O(N^2)$ tensor concatenation (`torch.cat`), measuring exact execution time in milliseconds.
- **`ollama_kv_cache_demo()`**: Demonstrates multi-turn conversation memory retention in local Ollama instance (`llama3.2:1b`), showing how turn 2 reuses historical KV cache.

### 2. Proprietary Cloud API Implementation: [proprietary_gemini.py](Inference-Optimization/01-kv-caching/proprietary_gemini.py)
- Demonstrates modern `google-genai` SDK context caching workflow:
  1. `client.caches.create()`: Pre-computes and uploads a system prompt context to Google Cloud TPUs.
  2. **Uncached Query**: Measures response latency when re-computing full context attention.
  3. **Cached Query**: Measures response latency when querying using `config=types.GenerateContentConfig(cached_content=...)`.
  4. **Performance Summary**: Prints latency comparison, speedup factor, and token usage metadata.

---

## 7. AWS & Hardware-Accelerated Deep Dive

- **AWS Inferentia2 / Trainium1**: AWS Neuron SDK requires explicit memory allocation for KV cache tensor shapes (`seq_len`). Neuron Core utilizes static KV cache buffers allocated at engine load time.
- **AWS Bedrock Prompt Caching**: Automatically detects matching prefix prompts (>1024 tokens) and reuses cached KV states across requests, lowering latency and cost.
- **SageMaker LMI**: Uses vLLM or TensorRT-LLM backends where KV-cache allocation can be fine-tuned via `gpu_memory_utilization` and `max_model_len` settings.

---

## 8. Advanced KV-Cache Optimization Techniques

1. **PagedAttention (vLLM)**: Allocates KV cache memory in non-contiguous physical blocks (pages), eliminating external memory fragmentation and increasing serving concurrency by up to $2\times - 4\times$.
2. **Quantized KV Caching**: Compresses KV cache tensors from FP16 (2 bytes) to INT8 (1 byte) or FP4/INT4 (0.5 bytes), saving up to 75% VRAM with minimal quality loss.
3. **Prefix Caching / Prompt Caching (RadixTree)**: Shared prompt prefixes across unrelated user requests are indexed automatically in a tree structure (e.g., SGLang, vLLM) so common prompt templates skip prefill computation.
