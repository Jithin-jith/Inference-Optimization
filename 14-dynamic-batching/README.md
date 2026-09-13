# Module 14: Dynamic Batching & Continuous Batching (Iteration-Level Scheduling)

## 1. Technical Explanation & Mathematical Intuition

In traditional **Static Batching**, requests arriving in a server queue are grouped into fixed-size batches (e.g., $B=4$).
However, because sequence generation lengths vary per request (Request 1 generates 20 tokens, Request 2 generates 300 tokens):
- Short requests complete early but are held back until the longest request finishes generation.
- GPU computational slots are filled with **wasted pad tokens**, reducing GPU utilization to under 20%.

### Continuous Batching (Iteration-Level Scheduling):
Engineered by Orca / vLLM:
- Batching operates at the **iteration level** (per single token step) rather than the request level.
- As soon as a request emits an `<EOS>` token, its slot is freed instantly and a new waiting request from the queue is inserted into the running batch **mid-generation**.

### Efficiency Gains
$$\text{GPU Utilization}_{\text{Continuous}} \approx 85\% - 95\% \quad \text{vs} \quad \text{GPU Utilization}_{\text{Static}} \approx 15\% - 30\%$$

---

## 2. Architecture & Execution Flow

```
Static Batching (Wasted GPU Slots):
Req 1 (Short): [Tok 1..20] [PAD] [PAD] [PAD] [PAD] ... (Waiting for Req 2)
Req 2 (Long) : [Tok 1.......................300]

Continuous Batching (Iteration-Level Slot Insertion):
Req 1 (Short): [Tok 1..20] -> FREED!
Req 3 (New)  :             -> INSERTED IMMEDIATELY AT TOKEN 21!
Req 2 (Long) : [Tok 1.......................300]
Zero padding waste! GPU running at 100% active capacity.
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **GPU Utilization** | Skyrockets GPU compute utilization to >85% | Complex scheduler queue management logic |
| **Inter-Token Latency** | Prevents long request starvation for short requests | Dynamic memory management (PagedAttention) required |
| **Serving Cost** | 3x to 5x higher global request concurrency per server | Latency jitter under peak load |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (vLLM / Triton Inference Server / TGI / Ollama)**:
  - **vLLM** and **TGI (Text Generation Inference)** enforce continuous iteration batching natively.
  - **Triton Inference Server** provides `dynamic_batching` configuration blocks in `config.pbtxt`.
- **Proprietary (Google Gemini Serving Clusters)**:
  - Google's global model serving microservices use continuous iteration scheduling across thousands of TPU nodes, seamlessly multiplexing queries from search, workspace, and Gemini API users.

---

## 5. AWS Deep Dive

- **SageMaker LMI Container**: Configured via `serving.properties`:
  ```properties
  option.max_rolling_batch_size=32
  option.rolling_batch=vllm
  ```
- **Triton on EC2 (`g5.12xlarge`)**: Uses Triton Dynamic Batcher configuration:
  ```protobuf
  dynamic_batching {
    max_queue_delay_microseconds: 5000
    preferred_batch_size: [ 4, 8, 16, 32 ]
  }
  ```
