"""
================================================================================
MODULE 06: EARLY EXIT DECODING & DYNAMIC LAYER HALTING
================================================================================

CONCEPT OVERVIEW:
-----------------
Standard transformer models process every token through ALL L stacked layers (e.g. 32 layers in Llama-3-8B).
However, simple syntax tokens (commas, articles, simple words) reach high classification confidence
at intermediate layers (e.g. layer 8 or 12).

EARLY EXIT DECODING MECHANISM:
------------------------------
Auxiliary classification heads (Exit Heads) are attached to intermediate hidden layers (L_1, L_2...).
At layer l, the model calculates the entropy of the intermediate token prediction:
Entropy(P_l) = - sum(P_l * log(P_l))

If Entropy(P_l) < Threshold:
- The model HALTS computation immediately at layer l.
- Skips upper L - l layers, saving 30% - 50% of computational FLOPs!

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `EarlyExitTransformerBlock`: PyTorch multi-layer neural block with intermediate Exit Heads,
   measuring output entropy and halting execution early on high-confidence tokens.
2. `ollama_adaptive_routing_demo()`: Query complexity router dispatching simple vs hard prompts.
================================================================================
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import ollama


class IntermediateExitHead(nn.Module):
    """
    Exit Head attached to an intermediate hidden layer.
    Computes vocabulary logits and calculates Shannon Entropy over prediction probabilities.
    """
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        logits = self.proj(x)
        probs = F.softmax(logits, dim=-1)
        # Compute Shannon Entropy: H(P) = -sum(P * log(P))
        # Low entropy = High confidence! High entropy = Uncertainty!
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return logits, entropy


class EarlyExitTransformerBlock(nn.Module):
    """
    Simulates a 12-layer Transformer backbone with Early Exit Heads at Layer 4 and Layer 8.
    """
    def __init__(self, hidden_dim, num_layers=12, vocab_size=500, exit_threshold=1.5):
        super().__init__()
        self.num_layers = num_layers
        self.exit_threshold = exit_threshold
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        # Attach intermediate exit heads at Layer 4 and Layer 8
        self.exit_head_4 = IntermediateExitHead(hidden_dim, vocab_size)
        self.exit_head_8 = IntermediateExitHead(hidden_dim, vocab_size)
        self.final_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            
            # --- Check Early Exit Condition at Layer 4 ---
            if i == 3:
                logits, entropy = self.exit_head_4(h)
                mean_entropy = entropy.mean().item()
                if mean_entropy < self.exit_threshold:
                    # Halt computation! Skip layers 5 through 12!
                    return logits, i + 1, mean_entropy, "EARLY_EXIT_LAYER_4"

            # --- Check Early Exit Condition at Layer 8 ---
            if i == 7:
                logits, entropy = self.exit_head_8(h)
                mean_entropy = entropy.mean().item()
                if mean_entropy < self.exit_threshold:
                    # Halt computation! Skip layers 9 through 12!
                    return logits, i + 1, mean_entropy, "EARLY_EXIT_LAYER_8"

        # --- Final Exit at Layer 12 ---
        logits = self.final_head(h)
        return logits, self.num_layers, 0.0, "FULL_EXIT_LAYER_12"


def run_pytorch_early_exit_demo():
    """
    Executes PyTorch Early Exit Transformer simulation on simple vs complex inputs.
    """
    print("=" * 70)
    print("1. PyTorch Dynamic Layer Early-Exit Simulation")
    print("=" * 70)

    hidden_dim = 128
    vocab_size = 500
    model = EarlyExitTransformerBlock(hidden_dim=hidden_dim, num_layers=12, vocab_size=vocab_size, exit_threshold=2.0)

    # Sample A: High-magnitude embeddings -> Low Entropy -> Early Exit at Layer 4!
    torch.manual_seed(10)
    simple_input = torch.randn(1, 10, hidden_dim) * 3.0
    logits, exited_at, entropy, status = model(simple_input)
    print(f"Sample A (High Confidence): Exited at Layer {exited_at}/12 | Status: {status} | Entropy: {entropy:.4f}")

    # Sample B: Low-magnitude embeddings -> High Entropy -> Requires Full 12 Layers!
    torch.manual_seed(42)
    complex_input = torch.randn(1, 10, hidden_dim) * 0.1
    logits, exited_at, entropy, status = model(complex_input)
    print(f"Sample B (Low Confidence):  Exited at Layer {exited_at}/12 | Status: {status} | Entropy: {entropy:.4f}")
    print("-" * 70 + "\n")


def ollama_adaptive_routing_demo():
    """
    Demonstrates query complexity classification and adaptive model selection.
    """
    print("=" * 70)
    print("2. Ollama Practical Demonstration: Query Complexity Classifier")
    print("=" * 70)

    sample_queries = [
        "What is the capital of France?",
        r"Write an $O(N \log N)$ algorithm in Rust to solve the traveling salesperson problem with dynamic programming."
    ]

    for q in sample_queries:
        word_count = len(q.split())
        if word_count < 8 and "algorithm" not in q:
            route = "LIGHT_FAST_MODEL (Llama-3.2-1B)"
        else:
            route = "HEAVY_PRO_MODEL (Llama-3.2-3B)"

        print(f"Query: '{q}'")
        print(f" -> Complexity Score: {'SIMPLE' if 'LIGHT' in route else 'COMPLEX'}")
        print(f" -> Routed to Target: {route}\n")


if __name__ == "__main__":
    run_pytorch_early_exit_demo()
    ollama_adaptive_routing_demo()
