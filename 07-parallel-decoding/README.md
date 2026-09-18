# Module 07: Parallel Decoding (Medusa, Eagle & Lookahead Decoding)

## 1. Technical Explanation & Mathematical Intuition

Standard autoregressive decoding generates tokens strictly one by one in sequence:
$$t_1 \to t_2 \to t_3 \to \dots \to t_N \quad \implies N \text{ sequential forward passes}$$

**Parallel Decoding** (such as **Medusa** or **Lookahead Decoding**) modifies the architecture or decoding algorithm to predict and verify multiple future tokens ($t_{k+1}, t_{k+2}, \dots, t_{k+K}$) simultaneously in a single forward pass without requiring a separate draft model.

### Medusa Architecture:
- Multiple auxiliary decoding heads (**Medusa Heads**) are added on top of the original model's last hidden state $h_t$.
- Head 1 predicts token $t+1$.
- Head 2 predicts token $t+2$.
- Head $k$ predicts token $t+k$.
- Generates a **tree of candidate token sequences**.
- In the next step, a single parallel forward pass with a customized attention mask verifies all tree paths simultaneously.

---

## 2. Architecture & Execution Flow

```
Standard Autoregressive (1 Token per Pass):
Step 1: Forward Pass -> [T1]
Step 2: Forward Pass -> [T2]
Step 3: Forward Pass -> [T3]

Medusa Parallel Decoding (3+ Tokens per Pass):
Hidden State h_t ---> Medusa Head 1 ---> Predict T1
                 ---> Medusa Head 2 ---> Predict T2
                 ---> Medusa Head 3 ---> Predict T3
                               |
                   Tree-Attention Parallel Verification
                               |
               [T1, T2, T3] Accepted in ONE single pass! (3x speedup)
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Speedup** | **1.8x to 2.8x faster decoding** | Requires training or attaching Medusa head parameters |
| **No Draft Model** | Eliminates secondary draft model memory overhead | Larger tree search space consumes extra VRAM |
| **Integrity** | Preserves original target model capability | Token acceptance rates vary by domain (code vs poetry) |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (vLLM / TensorRT-LLM / SGLang / PyTorch)**:
  - **Medusa**, **Eagle**, and **Lookahead Decoding** are supported in production engines like vLLM and SGLang.
  - Auxiliary linear heads (`MedusaHeadBlock`) attach to the base transformer backbone hidden state $h_t$ to predict candidate tokens ($t+1, t+2, t+3$), followed by parallel tree-attention verification.
  - vLLM `--speculative-decoding-type medusa` loads pre-trained Medusa heads directly onto base models (e.g. Llama-3-8B-Medusa).
- **Proprietary (Google Gemini Single-Request Parallel Decoding)**:
  - **Single-Request Parallel Branch Decoding**: Passing `candidate_count = N` inside **ONE single generation request** (`client.models.generate_content`) triggers Gemini's internal server-side multi-head decoder to branch candidate tokens ($c_1, c_2, \dots, c_N$) in parallel at each step of the autoregressive decode phase within a single inference graph execution.
  - Generates $N$ candidate completions concurrently inside the single request in $\sim 1\times$ single-request wall clock latency ($\approx 1.75\text{s}$ vs $5.07\text{s}$ for 3 serial requests, achieving a $2.89\times$ latency speedup).

---

## 5. AWS Deep Dive

- **SageMaker LMI (vLLM / TensorRT-LLM Engine)**: Supports Medusa heads via `option.speculative_draft_model` pointing to Medusa head weights on S3 or HuggingFace.
- **AWS Neuron SDK**: Neuron compiler (`neuronx-cc`) optimizes multi-head tree attention validation graphs for low-latency batch execution.
