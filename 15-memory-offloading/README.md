# Module 15: Memory Offloading (CPU / GPU / NVMe Layer Swapping)

## 1. Technical Explanation & Mathematical Intuition

When a large model exceeds physical GPU VRAM and purchasing extra GPU hardware is not viable, **Memory Offloading** enables running the model by partitioning and dynamically swapping model weights, KV-caches, or hidden states across a hierarchical storage tier:

1. **GPU VRAM** (~2 TB/s bandwidth - Fastest)
2. **Host System CPU RAM** (~100 GB/s bandwidth over PCIe Gen4/Gen5)
3. **NVMe SSD Storage** (~7 GB/s bandwidth - Slowest)

### DeepSpeed-ZeRO-Offload & llama.cpp Layer Offloading
For a model with $L$ layers:
- $L_{\text{gpu}}$ layers are loaded onto GPU VRAM.
- $L_{\text{cpu}}$ layers reside in System CPU RAM.
- During forward pass execution, layers are loaded dynamically from CPU RAM to GPU SRAM just-in-time over the PCIe bus.

### Bandwidth Bottleneck Formula
$$\text{Transfer Latency} = \frac{\text{Weight Parameters (Bytes)}}{\text{PCIe Bus Bandwidth (Bytes/sec)}}$$
For example, transferring a 5B parameter layer (10 GB in FP16) over PCIe Gen4 x16 (32 GB/s):
$$\text{Transfer Latency} = \frac{10 \text{ GB}}{32 \text{ GB/s}} = 0.3125 \text{ seconds per layer}$$

---

## 2. Architecture & Execution Flow

```
Hierarchical Memory Tier:

+-------------------------------------------------------------+
| GPU VRAM (High Bandwidth: 2.0 TB/s)  -> Active Layers 1..10  |
+-------------------------------------------------------------+
                              ^
                       PCIe Gen4 / Gen5 Bus (32-64 GB/s)
                              v
+-------------------------------------------------------------+
| System CPU RAM (Bandwidth: 100 GB/s) -> Layers 11..30       |
+-------------------------------------------------------------+
                              ^
                       NVMe Interface (7 GB/s)
                              v
+-------------------------------------------------------------+
| NVMe SSD Storage (Slowest)           -> Excess Weights/Cache|
+-------------------------------------------------------------+
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Zero Hardware Upgrade** | Runs 70B models on single 16GB/24GB GPUs | Generation speed is bottlenecked by PCIe bus bandwidth |
| **Cost Savings** | Slashes server infrastructure costs | Higher Inter-Token Latency (ITL) |
| **Use Case Fit** | Ideal for developer workstations, batch scoring, background tasks | Unsuitable for sub-second real-time chatbots |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (llama.cpp / Ollama / DeepSpeed ZeRO-Offload / Accelerate)**:
  - **Ollama / llama.cpp**: Allows setting GPU layer offloading flags (e.g. `--num-gpu 20` layers on GPU, remainder on CPU).
  - **HuggingFace Accelerate**: Provides `device_map="auto"` with `offload_folder="./offload"` for NVMe disk offloading.
- **Proprietary (Google Managed Serverless Abstraction)**:
  - Proprietary APIs (Gemini) hide underlying host/device memory boundaries entirely behind managed endpoints.

---

## 5. AWS Deep Dive

- **AWS EC2 Spot Instances**: Pair single-GPU instances (`g5.2xlarge` with 24GB VRAM) with high-speed NVMe instance store volumes (`i3en` or `g5` local NVMe) to offload 70B models cost-effectively.
- **DeepSpeed on SageMaker**: DeepSpeed ZeRO-3 Offload configured in `deepspeed_config.json`:
  ```json
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
  ```
