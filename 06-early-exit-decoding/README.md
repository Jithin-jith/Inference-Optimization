# Module 06: Early Exit Decoding & Model Routing

## 1. Technical Explanation & Mathematical Intuition

Standard transformer language models process every input token through all $L$ stacked layers (e.g., 32 layers in Llama-3-8B, 80 layers in 70B models), regardless of token complexity.

However, many simple syntax tokens (e.g., commas, articles, common words like "the", "and") can be predicted with high confidence at lower hidden layers (e.g., layer 8 or 12).

### Early Exit Mechanism:
Auxiliary classification heads (called **Exit Heads**) are attached to intermediate hidden layers $l \in \{L_1, L_2, \dots, L_k\}$.
At layer $l$, the model computes entropy / confidence score over the intermediate vocabulary projection $P_l(y \mid x)$:
$$\text{Entropy}(P_l) = -\sum_{v \in V} P_l(v) \log P_l(v)$$

If $\text{Entropy}(P_l) < \epsilon$ (where $\epsilon$ is a pre-defined threshold):
- The model **halts computation immediately** at layer $l$.
- Skips remaining $L - l$ upper layers, saving FLOPs and reducing per-token latency.

---

## 2. Architecture & Execution Flow

```
Input Token ---> Layer 1 -> Layer 2 -> ... -> Layer 8 (Exit Head Check)
                                                    |
                                       Entropy Check: < Threshold?
                                        /                        \
                                     [YES]                       [NO]
                                       |                           |
                       EXIT IMMEDIATELY! (Skip L9-L32)    Continue to Layer 9 -> ... -> Layer 32
                       Latency Savings: 65%               Full Transformer Execution
```

---

## 3. Production Trade-offs

| Aspect | Advantage | Disadvantage |
|---|---|---|
| **Latency Reduction** | Slashes execution time by 30% to 50% on simple tokens | Extra memory needed for intermediate exit head projection weights |
| **Compute Efficiency** | Dynamically scales FLOPs based on input complexity | Risk of quality loss on complex reasoning if threshold is too loose |
| **Adaptive Routing** | Pairs with API cascading (Flash vs Pro routing) | Requires fine-tuning or exit-head training |

---

## 4. Open-Source vs Proprietary Paradigm

- **Open-Source (PyTorch / HuggingFace / DeepSpeed)**:
  - Frameworks like **CALM** (Confident Adaptive Language Modeling) add exit heads to transformer layers.
  - Can be simulated using confidence scoring across model layers in PyTorch.
- **Proprietary (Google Gemini Adaptive Routing)**:
  - Enterprise API systems implement **Dynamic Model Cascade Routing**: simple queries (e.g. classification, extraction) are routed to fast models (`gemini-2.5-flash`), while complex reasoning queries (code, math proof) escalate to `gemini-2.5-pro`.

---

## 5. AWS Deep Dive

- **AWS Bedrock Model Routing / Guardrails**: Bedrock allows establishing dynamic router lambda functions to evaluate user input complexity before dispatching queries to Titan Light vs Claude / Llama 70B models.
- **SageMaker Multi-Model Endpoints (MME)**: Hosts light and heavy models on a shared container endpoint, allowing dynamic early-exit routing based on query complexity.
