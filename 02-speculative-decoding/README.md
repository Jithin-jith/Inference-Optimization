# Module 02: Speculative Decoding

## 1. Technical Explanation & Mathematical Intuition

LLM token decoding is memory-bandwidth bound: for every single generated token, all billions of weight parameters must be fetched from GPU VRAM to compute logic for a single prompt token.

**Speculative Decoding** overcomes this bottleneck by using two models:
1. A small, lightweight **Draft Model** $M_{\text{draft}}$ (e.g., 1B-3B parameters).
2. A large, high-capability **Target Model** $M_{\text{target}}$ (e.g., 70B parameters).

### Algorithm Steps:
1. **Draft Generation**: $M_{\text{draft}}$ generates $\gamma$ speculative candidate tokens autoregressively: $x_1, x_2, \dots, x_\gamma$. This is fast because $M_{\text{draft}}$ is small.
2. **Parallel Target Verification**: $M_{\text{target}}$ runs a single forward pass over all $\gamma + 1$ tokens in parallel (taking compute-bound prefill efficiency rather than memory-bound decode).
3. **Rejection Sampling / Validation**:
   For each candidate token $i \in [1 \dots \gamma]$, accept the token with probability:
   $$P(\text{accept}) = \min\left(1, \frac{p_{\text{target}}(x_i)}{p_{\text{draft}}(x_i)}\right)$$
   If candidate $k$ is rejected, stop draft acceptance, sample a replacement token from $M_{\text{target}}$, and discard remaining candidates.

### Speedup Formula
If $\alpha$ is the average acceptance rate of draft tokens ($\alpha \in [0, 1]$), the expected speedup factor $S$ is:
$$S = \frac{1 + \alpha \cdot \gamma}{1 + \frac{\text{Time}(M_{\text{draft}}) \times \gamma}{\text{Time}(M_{\text{target}})}}$$

---

## 2. Architecture & Execution Flow

```
Draft Model (Fast):   [T1] -> [T2] -> [T3] -> [T4] (Generates 4 candidate tokens)
                                 |
                                 v
Target Model (Parallel): Verify [T1, T2, T3, T4] in ONE single forward pass!
                                 |
Validation Check:               [OK] [OK] [FAIL] [DISCARD]
                                 |
Accepted Tokens:               T1, T2 + New Target Token T3' (3 tokens generated in ~1 target step time)
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Latency (ITL)** | 2x - 3x latency reduction for target model | Requires extra GPU memory to load draft model |
| **Output Quality** | Mathematically identical to target model distribution | Compute efficiency drops if draft acceptance rate is low (<50%) |
| **Hardware Fit** | Best for large 70B+ models | Not beneficial when GPU compute is already at 100% saturation |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (vLLM / Ollama / PyTorch)**:
  - vLLM supports speculative decoding using `--speculative-model <draft_model>` parameter.
  - Draft model must share the exact same vocabulary/tokenizer as target model (e.g., Llama-3-8B draft for Llama-3-70B target).
- **Proprietary (Google Gemini API)**:
  - Proprietary APIs implement speculative decoding internally in their serving stacks (e.g. Gemini 2.5 Pro using Gemini 2.5 Flash as a speculative draft engine).
  - API users can also emulate speculative routing by attempting draft generation with fast models (`gemini-2.5-flash`) and triggering fallback verification on heavy models (`gemini-2.5-pro`).

---

## 5. AWS Deep Dive

- **SageMaker LMI (Large Model Inference)**: Supports vLLM and TensorRT-LLM spec decoding backends. Configurable via `option.speculative_draft_model`.
- **AWS Inferentia2 (INF2)**: Speculative decoding can be compiled directly onto AWS Neuron cores by loading both draft `.neff` and target `.neff` models onto adjacent NeuronCores.
- **AWS Bedrock**: Built-in spec decoding inside Amazon Titan and Llama 3 models hosted on Bedrock provisioned throughput instances.
