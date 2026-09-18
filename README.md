# Master Learning Repository: LLM Inference Optimization

Welcome to the comprehensive, production-grade learning repository on **LLM Inference Optimization Techniques**. This repository provides complete technical deep dives, mathematical intuition, production trade-offs, AWS cloud deployment guides, and runnable Python source code implementations for 16 core optimization techniques across both **Open-Source (Ollama / vLLM / PyTorch)** and **Proprietary (Google Gemini API)** paradigms.

> 💡 **For Managers & Executives**: Check out [`TECHNICAL_DOCUMENTATION.md`](file:///f:/Projects/Inference-Optimization/TECHNICAL_DOCUMENTATION.md) for a high-level summary, decision matrices, metrics breakdown, and ROI guide.

---

## 📚 Course Structure & Modules

The repository is organized into 16 self-contained modules, each covering a specific optimization strategy:

| Module | Optimization Technique | Focus Area | Code Implementations |
|---|---|---|---|
| [01-kv-caching](./01-kv-caching/README.md) | **KV-Caching** | Memory & Attention | `opensource_ollama.py`, `proprietary_gemini.py` |
| [02-speculative-decoding](./02-speculative-decoding/README.md) | **Speculative Decoding** | Decoding Acceleration | `opensource_ollama.py`, `proprietary_gemini.py` |
| [03-flash-attention](./03-flash-attention/README.md) | **FlashAttention** | Memory I/O & SRAM | `opensource_ollama.py`, `proprietary_gemini.py` |
| [04-paged-attention](./04-paged-attention/README.md) | **PagedAttention** | VRAM Virtualization | `opensource_ollama.py`, `proprietary_gemini.py` |
| [05-batch-inference](./05-batch-inference/README.md) | **Batch Inference** | Throughput Maximization | `opensource_ollama.py`, `proprietary_gemini.py` |
| [06-adaptive-model-cascading](./06-adaptive-model-cascading/README.md) | **Adaptive Model Cascading** | Dynamic Model Routing | `opensource_ollama.py`, `proprietary_gemini.py` |
| [07-parallel-decoding](./07-parallel-decoding/README.md) | **Parallel Decoding** | Multi-Token Prediction | `opensource_ollama.py`, `proprietary_gemini.py` |
| [08-mixed-precision-inference](./08-mixed-precision-inference/README.md) | **Mixed Precision Inference** | FP16/BF16/FP8 Quantization | `opensource_ollama.py`, `proprietary_gemini.py` |
| [09-quantized-kernels](./09-quantized-kernels/README.md) | **Quantized Kernels** | GGUF/AWQ/GPTQ Kernels | `opensource_ollama.py`, `proprietary_gemini.py` |
| [10-tensor-parallelism](./10-tensor-parallelism/README.md) | **Tensor Parallelism** | Intra-Layer GPU Sharding | `opensource_ollama.py`, `proprietary_gemini.py` |
| [11-pipeline-parallelism](./11-pipeline-parallelism/README.md) | **Pipeline Parallelism** | Inter-Layer Stage Sharding | `opensource_ollama.py`, `proprietary_gemini.py` |
| [12-sequence-parallelism](./12-sequence-parallelism/README.md) | **Sequence Parallelism** | Ring-Attention & Long Context | `opensource_ollama.py`, `proprietary_gemini.py` |
| [13-graph-optimization](./13-graph-optimization/README.md) | **Graph Optimization** | ONNX / TensorRT Fused Kernels | `opensource_ollama.py`, `proprietary_gemini.py` |
| [14-dynamic-batching](./14-dynamic-batching/README.md) | **Dynamic Batching** | Continuous Request Scheduling | `opensource_ollama.py`, `proprietary_gemini.py` |
| [15-memory-offloading](./15-memory-offloading/README.md) | **Memory Offloading** | GPU/CPU/NVMe Layer Swapping | `opensource_ollama.py`, `proprietary_gemini.py` |
| [16-streaming-generation](./16-streaming-generation/README.md) | **Streaming Generation** | Token Latency Reduction | `opensource_ollama.py`, `proprietary_gemini.py` |
| [17-early-exit-decoding](./17-early-exit-decoding/README.md) | **Early Exit Decoding** | Dynamic Layer Halting | `opensource_ollama.py`, `proprietary_gemini.py` |

---

## 🛠️ Prerequisites & Setup

### 1. Environment Setup
Clone the repository and install the Python dependencies:
```bash
git clone https://github.com/your-org/Inference-Optimization.git
cd Inference-Optimization
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Open-Source Setup (Ollama)
Install [Ollama](https://ollama.com/) locally and pull the recommended open-source models:
```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```

### 3. Proprietary Setup (Google Gemini API)
Obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/) and set your environment variable:
```bash
# On Windows PowerShell:
$env:GOOGLE_API_KEY="your-gemini-api-key"

# On Linux/macOS:
export GOOGLE_API_KEY="your-gemini-api-key"
```

---

## 🚀 Running Code Examples

Each module directory contains two standalone Python scripts:

1. **Open-Source Implementation (`opensource_ollama.py`)**:
   Runs local open-weights inference using the official `ollama` client or native PyTorch/vLLM primitives.
   ```bash
   python 01-kv-caching/opensource_ollama.py
   ```

2. **Proprietary API Implementation (`proprietary_gemini.py`)**:
   Interfaces with Google Gemini using the modern `google-genai` SDK, showcasing API features like Context Caching, Batch API, Adaptive Routing, and Streaming.
   ```bash
   python 01-kv-caching/proprietary_gemini.py
   ```

---

## ☁️ AWS Integration Coverage

Every module README includes a dedicated **AWS Deep Dive** section detailing:
- **AWS Neuron SDK & Hardware**: How to compile and serve models on AWS Inferentia (INF1/INF2) and Trainium (TRN1).
- **SageMaker LMI (Large Model Inference)**: Pre-configured vLLM/TensorRT-LLM container configurations.
- **AWS Bedrock**: Provisioned Throughput, Bedrock Prompt Caching, and serverless LLM optimization strategies.

---

## 📜 License
MIT License. Free for educational and commercial learning.