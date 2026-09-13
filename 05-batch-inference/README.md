# Module 05: Batch Inference & Batch APIs

## 1. Technical Explanation & Mathematical Intuition

Single-request LLM inference suffers from low GPU compute saturation because loading model weights $W \in \mathbb{R}^{d_1 \times d_2}$ for a single input vector $x \in \mathbb{R}^{1 \times d_1}$ requires high memory bandwidth transfers with minimal matrix multiplication (GEMM) operations.

### Arithmetic Intensity Formula
$$\text{Arithmetic Intensity} = \frac{\text{Total FLOPs}}{\text{Total Bytes Transferred from HBM}}$$

For batch size $b = 1$:
$$\text{Arithmetic Intensity} \approx 1 \text{ FLOP / Byte} \quad \implies \text{Severe Memory Bandwidth Bottleneck}$$

When batching $b$ requests together ($X \in \mathbb{R}^{b \times d_1}$):
$$\text{Arithmetic Intensity} \approx b \text{ FLOPs / Byte} \quad \implies \text{Transitions toward GPU Compute Saturation}$$

By increasing batch size $b$, the cost of fetching weights from GPU VRAM is amortized across $b$ tokens, increasing global **Token Throughput (tokens/second)** by order of magnitude.

---

## 2. Architecture & Execution Flow

```
Sequential Single-Request (Batch Size = 1):
Request 1: Load Weights -> Compute Token -> Output  (Latency: T)
Request 2: Load Weights -> Compute Token -> Output  (Latency: T)
Request 3: Load Weights -> Compute Token -> Output  (Latency: T)
Total Time: 3T | GPU Compute Core Utilization: ~15%

Batched Multi-Request (Batch Size = 3):
[Request 1, Request 2, Request 3] -> Load Weights ONCE -> Compute 3 Tokens in Parallel
Total Time: ~1.2T | GPU Compute Core Utilization: ~85% | Speedup: 2.5x Throughput
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Global Throughput** | Up to **5x to 10x higher throughput** | Per-request TTFT increases slightly due to batch assembly wait |
| **Operational Cost** | 50% lower cost per processed million tokens | High VRAM memory footprint for aggregated KV caches |
| **Use Case Fit** | Ideal for offline data processing, evaluation, synthetic data | Not suited for ultra-low-latency real-time voice apps |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (PyTorch DataLoader / vLLM Batching)**:
  - In standard PyTorch, inputs are padded to maximum sequence length using pad tokens and processed as batched 2D/3D tensors.
  - Modern engines use dynamic batching queues (e.g. Triton Inference Server or vLLM engines).
- **Proprietary (Google Gemini Batch API)**:
  - Gemini provides an explicit **Batch API** endpoint designed for non-real-time jobs.
  - Submitting batch jobs to Gemini offers **50% discount** on API costs and higher rate limits compared to standard real-time endpoints.

---

## 5. AWS Deep Dive

- **AWS Bedrock Batch Inference**: Allows submitting bulk dataset jobs stored in S3. Results are written asynchronously back to S3 at a lower cost tier.
- **AWS SageMaker Batch Transform**: SageMaker feature for running offline batch inference pipelines over large Amazon S3 datasets using multi-GPU instances.
- **AWS Inferentia2 (INF2)**: Neuron compiler supports static batch sizes (e.g. `--batch-size 8` or `16`), optimizing Tensor Engine pipeline throughput.
