# Module 05: Batch Inference & Batch APIs

## Overview
**Batch Inference** is the process of grouping multiple independent requests together into a single compute pass so that the GPU can reuse model weights far more efficiently, dramatically increasing throughput and lowering cost per token.

---

## 1. Why Batch Size Matters

Suppose a neural network layer has a weight matrix:

$$W \in \mathbb{R}^{d_1 \times d_2}$$

For a single input request ($x \in \mathbb{R}^{1 \times d_1}$), the linear transformation is:

$$y = xW$$

With **Batch Size = 1**, the GPU must fetch the entire weight matrix $W$ (gigabytes of data) from memory to compute a single matrix-vector product.

With **Batch Size = 8** ($X \in \mathbb{R}^{8 \times d_1}$):

$$Y = XW$$

Now, the exact same weight matrix $W$ is loaded once into GPU compute registers and reused across all 8 input vectors simultaneously.

```
Batch Size = 1 (Weight Fetch Waste):
Load W from HBM ---> Process Request 1
Load W from HBM ---> Process Request 2
Load W from HBM ---> Process Request 3

Batch Size = 3 (Amortized Weight Reuse):
Load W from HBM ONCE
   │
   ├──> Process Request 1
   ├──> Process Request 2
   └──> Process Request 3
```

---

## 2. Arithmetic Intensity & The Memory Bottleneck

The fundamental metric governing GPU efficiency is **Arithmetic Intensity**:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved from HBM}}$$

- **FLOPs**: Total floating-point math operations performed by GPU Tensor Cores.
- **HBM**: GPU High-Bandwidth Memory where model weights and KV-caches reside.

When batch size is small, the GPU spends most of its clock cycles fetching weights from HBM (~2–3 TB/s) rather than computing (~300–1000 TFLOPS). This leaves the GPU severely **memory-bandwidth bound**.

> [!NOTE]
> **Warehouse Worker Analogy**: Imagine a warehouse worker who must walk to the far side of a warehouse to fetch a tool. If they walk all that way to use the tool on a single screw, they spend 99% of their time walking. Batching gives the worker 8 or 64 screws to drive per trip, maximizing active work time.

---

## 3. Example: Real-World LLM Serving

Suppose 8 users send prompts concurrently:
- User 1: *"What is AI?"*
- User 2: *"Explain Python"*
- User 3: *"What is RAG?"*
- ...
- User 8: *"Explain Docker"*

```
Without Batching (Sequential Bottleneck):
Request 1 ──> GPU GEMM ──> Response 1
Request 2 ──> GPU GEMM ──> Response 2
Request 3 ──> GPU GEMM ──> Response 3
...

With Batching (Parallel Tensor Compute):
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼──> Batched GEMM (X · W) ──> GPU Tensor Cores ──> 8 Responses
...       ─┤
Request 8 ─┘
```

Batching aggregates smaller matrix-vector operations into large matrix-matrix multiplications, achieving significantly higher **Token Throughput**:

$$\text{Throughput} = \frac{\text{Total Tokens Processed}}{\text{Second}}$$

---

## 4. Latency vs. Throughput

A critical architectural distinction is that **batching increases throughput, not per-request latency**.

| Mode | Request Latency | System Throughput |
|---|---|---|
| **Batch Size = 1** | 100 ms | $\approx 10 \text{ requests/sec}$ |
| **Batch Size = 8** | 150 ms | $\approx 53 \text{ requests/sec}$ ($8 \text{ req} / 0.15 \text{ s}$) |

While individual requests take slightly longer (150 ms vs 100 ms), global throughput increases by **>5x**. 

### Ideal Production Use Cases
Batching is essential for offline or asynchronous workloads:
- Offline document processing & PDF extraction
- Large-scale embedding generation
- Offline LLM evaluation & benchmarking
- Synthetic dataset generation
- Bulk document summarization & categorization

---

## 5. Static vs. Dynamic Batching

```
Static Batching (Fixed Queue Wait):
Queue: [R1, R2, R3] ──> Waits for R4..R8 ──> [Idle GPU Wait!]

Dynamic / Continuous Batching (vLLM / Triton):
R1 ─┐
R2 ─┼──> Adaptive Time Window (e.g. 5ms) ──> Immediate GPU Dispatch
R3 ─┘
```

- **Static Batching**: Fixes a strict `batch_size = 8`. If only 3 requests arrive, the engine waits, causing GPU stall.
- **Dynamic Batching**: Modern inference servers (NVIDIA Triton, vLLM) collect requests over a configurable micro-window (e.g., 5 ms) or dynamically inject new requests at token iteration steps, maximizing utilization without artificial waiting.

---

## 6. Why LLM Batching is Complicated (Padding Waste)

LLMs generate tokens autoregressively, and different user requests have varying sequence lengths:
- **Request A**: 100 tokens
- **Request B**: 500 tokens
- **Request C**: 50 tokens

Naive tensor batching requires padding sequences to match the longest request:
```
Request A: ██████████░░░░░░░░░░ (100 tokens + padding)
Request B: ████████████████████ (500 tokens)
Request C: █████░░░░░░░░░░░░░░░ (50 tokens + padding)
```
*(where ░ represents wasted compute on padding tokens)*

Modern inference engines overcome this waste using **Iteration-Level / Continuous Batching** (e.g., Orca, vLLM) and **PagedAttention**, dynamic eviction/injection of requests at each decoding step.

---

## 7. Batch API vs. Inference Engine Batching

It is important to distinguish internal engine batching from cloud provider **Batch APIs**:

```
1. Inference Engine Batching (Real-Time):
   Online Users ──> Real-Time HTTP Requests ──> Dynamic Queue ──> vLLM GPU Engine

2. Provider Batch API (Offline Asynchronous):
   100,000 Prompts File ──> Upload to Batch API ──> Asynchronous TPU/GPU Job ──> 50% Cost Discount
```

---

## 8. Real-World AWS Architecture Example

Consider extracting metadata (Customer, Country, Dates) from **1,000,000 contracts** stored in Amazon S3:

```
Real-Time API (Inefficient & Expensive):
Contract 1..1M ──> Individual HTTP Requests ──> High Latency & Full Cost Rate

AWS S3 Batch Workflow (Cost-Optimized):
S3 Input Bucket (1M Contracts JSONL)
       │
       ▼
AWS Bedrock / SageMaker Batch Inference Job
       │
       ▼
Asynchronous LLM Processing (50% Cost Discount)
       │
       ▼
S3 Output Bucket (Results JSONL)
```

---

## 9. Cost Intuition

$$\text{Poor GPU Utilization} \implies \text{More GPU Execution Time} \implies \text{Higher Cost per Token}$$

$$\text{High Batch Utilization} \implies \text{Maximum Tokens / GPU-Second} \implies \text{Lower Cost per Token}$$

By amortizing fixed weight loading costs across thousands of parallel requests, inference providers offer **50% discounts** on Batch API endpoints compared to standard real-time endpoints.

---

## 10. Key Takeaway & Architectural Mental Model

```
                             BATCH INFERENCE
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
               Higher Throughput             Better GPU Usage
                     │                             │
                     ▼                             ▼
              More Requests/Sec          Amortize Weight Loading
                     │
                     ▼
             Lower Cost/Request
```

The core objective of batching in LLM optimization is shifting from:
> *"Load weights $\rightarrow$ do a little work $\rightarrow$ load weights again"*

To:
> *"Load/use weights efficiently $\rightarrow$ perform maximum GEMM computation $\rightarrow$ maximize tokens/second."*

---

## 11. Benchmark Insights & Common Misconceptions

When running small benchmark scripts (e.g., 4 prompts), the output metrics can sometimes appear counter-intuitive:

### Misconception 1: "Batch API is Slower (133s vs. 6s)"
- **Why it happens in micro-benchmarks**: Cloud Batch APIs (`client.batches.create`) incur ~1–2 minutes of **queue scheduling overhead** while Google Cloud provisions background worker nodes. For 4 prompts, scheduling overhead dominates total wall-clock time.
- **Production Reality**: For **100,000 documents**:
  - **Sequential Mode ($b=1$)**: $100,000 \times 1.6\text{s} = 160,000\text{ s} \approx \mathbf{44.4\text{ hours}}$ (plus hitting `429 Rate Limit` crashes).
  - **Batch API Mode**: Processes all 100,000 prompts in parallel across global TPU Pods in **15–30 minutes**.

### Misconception 2: "Batch API Cost Appears Higher"
- **Why it happens in micro-benchmarks**: In small runs, the model may generate longer candidate text (e.g., 116 output tokens in batch vs. 12 output tokens in real-time), increasing total tokens generated.
- **Production Reality**: Cloud providers bill Batch APIs at a **50% Unit Price Discount** ($0.5\times$ rate per 1M tokens). For 1 Billion tokens, Real-Time costs $\$150.00$ while Batch API costs $\mathbf{\$75.00}$ (saving $\$75.00$).

### Decision Matrix: Real-Time vs. Batch API

```
                     IS YOUR WORKLOAD REAL-TIME?
                                 │
                 ┌───────────────┴───────────────┐
                 YES                             NO
                 │                               │
                 ▼                               ▼
       Synchronous API (b=1)            Batch Inference API
    - Real-time Voice / Chat         - 100,000 S3 PDF Contracts
    - Low TTFT (<200ms needed)       - Synthetic Data Generation
    - Instant interactive responses  - Offline LLM Benchmark Evaluation
                                     - 50% Token Price Discount
```

