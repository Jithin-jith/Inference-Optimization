# Module 04: PagedAttention (vLLM Engine Architecture)

## 1. Technical Explanation & Mathematical Intuition

In standard LLM inference serving frameworks, KV-cache memory for a request is allocated as a single, contiguous block of GPU VRAM based on the **maximum potential context length** (e.g., 2048 or 4096 tokens).

### Memory Waste in Standard Serving:
1. **Internal Fragmentation**: Reserved VRAM for max sequence length that is never generated (e.g., reserving 2048 slots when user prompt + response is only 200 tokens).
2. **External Fragmentation**: Virtual memory gaps between contiguous sequence allocations.
3. **Over-Reservation Waste**: Up to **60% to 80%** of GPU VRAM is wasted as pre-allocated, unused padding!

### PagedAttention Solution (Virtual Memory Paging):
Inspired by Virtual Memory in Operating Systems:
- KV-caches are broken into fixed-size **Physical KV Blocks** (e.g., 16 tokens per block).
- Memory is allocated **dynamically block-by-block** as new tokens are generated.
- A **Block Table** maps logical token positions to non-contiguous physical VRAM block addresses.

### Mathematical Memory Utilization
$$\text{Memory Efficiency} = \frac{\text{Actual Used KV Tokens}}{\text{Allocated Physical Blocks} \times \text{Block Size}} \approx 96\% - 99\%$$

---

## 2. Architecture & Execution Flow

```
Logical KV Cache (Sequence 1):
[Tok 0..15 (Block 0)] -> [Tok 16..31 (Block 1)] -> [Tok 32..47 (Block 2)]

OS-Style Block Table Mapping:
Logical Block 0  --->  Physical Block 7  (GPU VRAM Address 0x7F00)
Logical Block 1  --->  Physical Block 2  (GPU VRAM Address 0x2A00)
Logical Block 2  --->  Physical Block 14 (GPU VRAM Address 0xDF00)

Physical VRAM Layout (Non-Contiguous Allocation - ZERO Waste!):
[Phys Block 0 (Free)] [Phys Block 1 (Seq 2)] [Phys Block 2 (Seq 1 - Blk 1)] ... [Phys Block 7 (Seq 1 - Blk 0)]
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **GPU Utilization** | Increases serving concurrency by **2x to 4x** | Introduces slight block table pointer lookup overhead |
| **VRAM Waste** | Slashes memory waste from ~70% down to <4% | Fixed block size must be tuned (typically 16 or 32) |
| **Prefix Sharing** | Allows zero-copy KV cache sharing across parallel requests | Requires specialized C++/CUDA serving kernels (vLLM) |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (vLLM / TensorRT-LLM / Ollama)**:
  - **vLLM** pioneered PagedAttention. It manages a centralized memory pool of physical KV blocks on GPU VRAM.
  - Allows multiple user sessions (e.g. tree search, beam search, speculative decoding) to share physical KV blocks via copy-on-write reference counters.
- **Proprietary (Google Gemini API)**:
  - Google Cloud infrastructure uses serverless KV page block allocators across distributed TPU clusters.
  - Enables Gemini to serve millions of concurrent requests seamlessly without out-of-memory crashes.
  - Demonstrated in `proprietary_gemini.py` by benchmarking **Sequential Execution** (single-worker queuing) vs **Concurrent Asynchronous Execution** (PagedAttention page-table serving via `client.aio`).

---

## 5. Benchmark Comparison: Sequential vs. Concurrent Serving

In `proprietary_gemini.py`, we benchmark the performance impact of high-concurrency request dispatch (leveraging cloud PagedAttention page allocators) over identical prompts:

| Metric | Sequential Mode (Traditional Queue) | Concurrent Async Mode (PagedAttention) | Impact |
|---|---|---|---|
| **Execution Flow** | Requests processed linearly ($N \times T_{\text{req}}$) | Requests dispatched in parallel via HTTP/2 multiplexing | **3x – 4x Total Wall-Clock Time Reduction** |
| **VRAM Management** | Pre-allocates max contiguous KV slots per request | Dynamic allocation of 16-token physical KV page blocks | **Slashes memory fragmentation from ~70% to <4%** |
| **Throughput (req/s)** | Low (bottlenecked by single worker queue) | High (serves $N$ prompts simultaneously) | **Scales linearly with available physical page pool** |

```
Sequential Execution (Unpaged / Single Queue):
Req 1 ---> [Process & Allocate Max KV] ---> Complete (1.5s)
Req 2 ------------------------------------> [Process & Allocate Max KV] ---> Complete (3.0s)
Req 3 --------------------------------------------------------------------> [Process & Allocate Max KV] ---> Complete (4.5s)
Total Time: 4.5s

Concurrent Asynchronous Execution (PagedAttention Page Allocator):
Req 1 ---> [Block Table Page Allocator] \
Req 2 ---> [Block Table Page Allocator]  ===> Dispatch Parallel TPU Workers ---> All Complete (~1.5s Total)
Req 3 ---> [Block Table Page Allocator] /
```

---

## 6. AWS Deep Dive

- **AWS SageMaker LMI Containers**: Uses vLLM engine with PagedAttention enabled by default. Configured via `environment.json` or `serving.properties`:
  ```properties
  option.engine=Python
  option.model_id=meta-llama/Meta-Llama-3-8B
  option.gpu_memory_utilization=0.90
  option.paged_attention=true
  ```
- **AWS EC2 Multi-GPU (`g5.12xlarge`, `p4d.24xlarge`)**: PagedAttention scales linearly across multi-GPU nodes when combined with Tensor Parallelism.

