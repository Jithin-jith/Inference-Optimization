# Module 11: Pipeline Parallelism (PP)

## 1. Technical Explanation & Mathematical Intuition

While **Tensor Parallelism (TP)** splits individual weight matrices *inside* layers, **Pipeline Parallelism (PP)** divides total transformer layers sequentially *across* different GPU nodes or devices.

For a model with $L$ total layers distributed across $P$ pipeline stages (GPUs):
$$\text{Layers per GPU Stage} = \frac{L}{P}$$

For example, a 96-layer model split across 4 GPUs:
- **GPU 0 (Stage 1)**: Layers 1 to 24
- **GPU 1 (Stage 2)**: Layers 25 to 48
- **GPU 2 (Stage 3)**: Layers 49 to 72
- **GPU 3 (Stage 4)**: Layers 73 to 96

### Pipeline Bubble Problem & Micro-Batching
If processed naively, GPU 1 remains idle waiting for GPU 0 to finish layer 24.
To solve this, inputs are split into $M$ **micro-batches**. As GPU 0 finishes processing micro-batch 1, it passes intermediate activation tensors to GPU 1 via point-to-point network communication while GPU 0 immediately begins processing micro-batch 2.

### Pipeline Bubble Fraction Formula
$$\text{Bubble Fraction} = \frac{P - 1}{M + P - 1}$$
Where $P$ is the number of pipeline stages and $M$ is the number of micro-batches ($M \gg P$).

---

## 2. Architecture & Execution Flow

```
Pipeline Stages across 4 GPU Nodes:

Node 0 (GPU 0): [Layers 1..24]   ==Activations==>  Node 1 (GPU 1): [Layers 25..48]
                                                          ||
                                                     Activations
                                                          ||
                                                          \/
Node 3 (GPU 3): [Layers 73..96]  <==Activations==  Node 2 (GPU 2): [Layers 49..72]
       |
  Final Logits
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Multi-Node Scaling** | Operates effectively across standard Ethernet network links | Introduces pipeline idle "bubbles" |
| **VRAM Footprint** | Divides weight & KV VRAM evenly across multi-node clusters | Increases Time to First Token (TTFT) latency due to sequential stages |
| **Network Bandwidth** | Transfers only activation tensors ($B \times S \times d$) between stages | Requires $M \gg P$ micro-batches for high pipeline efficiency |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (DeepSpeed Pipeline / Megatron-LM / vLLM)**:
  - Supported via DeepSpeed `--pipeline-parallel-size` or vLLM `--pipeline-parallel-size`.
  - Uses PyTorch P2P point-to-point communication (`torch.distributed.send` / `recv`).
- **Proprietary (Google Gemini Multi-Cluster Pipelines)**:
  - Google dispatches giant Gemini 2.5 models across TPU Pod slices linked via Optical Circuit Switches (OCS) using pipeline execution.

---

## 5. AWS Deep Dive

- **AWS SageMaker LMI**: Supports Pipeline Parallelism combined with Tensor Parallelism (e.g. `TP=4, PP=2` on 8-GPU instances). Configured via `option.pipeline_parallel_degree=2`.
- **AWS Cluster Placement Groups**: When deploying PP across multiple EC2 instances (e.g., 2x `p4d.24xlarge` nodes), launch instances inside an **AWS Cluster Placement Group** with **EFA (Elastic Fabric Adapter)** 800 Gbps network interfaces to minimize inter-node transmission latency.
