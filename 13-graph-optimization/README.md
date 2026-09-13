# Module 13: Computational Graph Optimization (ONNX Runtime, TensorRT & Torch Compile)

## 1. Technical Explanation & Mathematical Intuition

In default Python frameworks (like eager PyTorch), every individual layer operation (e.g. `Linear` $\to$ `BiasAdd` $\to$ `LayerNorm` $\to$ `GeLU`) triggers an individual CUDA kernel launch overhead from the Host CPU to the GPU Device.

For deep networks running single-token decode passes, **Host CPU Python runtime overhead** can account for up to **30% to 50%** of total execution latency!

### Graph Compilation Solutions:
1. **Operator Fusion**: Fuses multiple adjacent mathematical ops into a single compiled CUDA kernel (e.g., combining `BiasAdd + LayerNorm + GeLU` into 1 GPU memory load and write).
2. **CUDA Graphs (`cudaGraphCapture`)**: Records the entire sequence of CUDA kernel launches once during warm-up. Subsequent inference iterations execute the entire graph on the GPU with **zero CPU launch overhead**.
3. **ONNX Runtime & TensorRT-LLM**: Compiles high-level computation graphs into optimized target hardware executables (`.engine` or `.onnx`).

---

## 2. Architecture & Execution Flow

```
Eager PyTorch Execution (High Overhead):
CPU ---> Launch GEMM Kernel ---> GPU Executes GEMM ---> GPU Writes Result to Memory
CPU ---> Launch Bias Add Kernel ---> GPU Executes Bias Add ---> GPU Writes Result to Memory
CPU ---> Launch LayerNorm Kernel ---> GPU Executes LayerNorm ---> GPU Writes Result to Memory

Compiled Graph (TensorRT / ONNX / Torch Compile):
CPU ---> Launch Single Fused Kernel Graph ONCE
GPU Executes [GEMM + BiasAdd + LayerNorm] inside fast SRAM tiles in ONE pass!
(20% - 40% Speedup, Zero CPU Overhead)
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Latency Reduction** | Slashes host CPU launch overhead to zero | Requires compilation step (warm-up build time) |
| **Operator Fusion** | Maximizes GPU SRAM register reuse | Static shapes required for optimal TensorRT speedup |
| **Throughput** | Boosts global token generation throughput by 20%-40% | Complex deployment build pipelines |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (ONNX Runtime / TensorRT-LLM / Torch Compile)**:
  - `torch.compile(model, mode="max-autotune")` compiles PyTorch models using OpenAI Triton kernels.
  - TensorRT-LLM compiles custom `.engine` files tailored to specific GPU models (e.g. H100 or A10G).
- **Proprietary (Google JAX / XLA Graph Compiler)**:
  - Google Gemini is compiled using **XLA (Accelerated Linear Algebra)** inside JAX, which automatically fuses matrix operations into high-efficiency TPU HBM/SRAM kernels.

---

## 5. AWS Deep Dive

- **AWS Neuron SDK (`neuronx-cc`)**: Model weights are compiled into Neuron Executable Format (`.neff`) files using XLA graph compiler optimizations before loading onto Inferentia2 or Trainium1.
- **SageMaker TensorRT-LLM Containers**: Supports pre-compiled TensorRT engines for ultra-low latency enterprise deployments.
