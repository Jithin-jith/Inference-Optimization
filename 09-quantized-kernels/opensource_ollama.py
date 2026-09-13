"""
================================================================================
MODULE 09: QUANTIZED KERNELS (GGUF, AWQ, GPTQ & INT8/INT4)
================================================================================

CONCEPT OVERVIEW:
-----------------
Quantization converts floating-point weights (FP16/BF16 - 16 bits) to low-precision integer formats
(INT8 - 8 bits, INT4 - 4 bits).

AFFINE QUANTIZATION FORMULA:
----------------------------
q = round(x / Scale) + ZeroPoint
Dequantized_x = Scale * (q - ZeroPoint)

Where Scale = (x_max - x_min) / (2^n - 1).

QUANTIZATION PARADIGMS:
-----------------------
1. GGUF (llama.cpp): Block-wise quantization format (Q4_K_M, Q5_K_M) optimized for CPU, Apple Metal, and GPUs.
2. AWQ (Activation-aware Weight Quantization): Protects the 1% most salient weight channels corresponding
   to large activation magnitudes, preserving 4-bit accuracy.
3. GPTQ (Post-Training Quantization): Uses second-order Hessian optimization to quantize layer weights.

KEY BENEFITS:
-------------
- 3.5x - 4x VRAM Reduction: Runs a 70B model in ~35GB VRAM instead of 140GB!
- 4x Bandwidth Compression: Slashes GPU memory bus bottleneck during token decode.

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `quantize_int8()` & `dequantize_int8()`: Structural PyTorch functions implementing affine INT8 quantization.
2. `torch_quantization_demo()`: Measures VRAM compression ratio and reconstruction MAE error.
3. `ollama_quantized_demo()`: Execution of local GGUF quantized models with Ollama.
================================================================================
"""

import time
import torch
# pyrefly: ignore [missing-import]
import ollama


def quantize_int8(weight_tensor: torch.Tensor):
    """
    Performs symmetric 8-bit integer (INT8) quantization on a floating point tensor.
    Maps range [-abs_max, +abs_max] to integer range [-127, +127].
    """
    x_max = torch.max(torch.abs(weight_tensor))
    scale = x_max / 127.0
    scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
    
    # Quantize: Float -> INT8
    quantized_int8 = torch.round(weight_tensor / scale).to(torch.int8)
    return quantized_int8, scale


def dequantize_int8(quantized_tensor: torch.Tensor, scale: torch.Tensor):
    """
    Dequantizes INT8 representation back to floating point FP32 format for matrix math.
    """
    return quantized_tensor.to(torch.float32) * scale


def torch_quantization_demo():
    """
    Measures PyTorch memory compression ratio and MAE error between original FP32 and INT8.
    """
    print("=" * 70)
    print("1. PyTorch 8-Bit Weight Quantization & Compression Simulation")
    print("=" * 70)

    torch.manual_seed(42)
    # Generate 1000x1000 weight matrix in FP32 format (4 MB)
    original_weights = torch.randn(1000, 1000, dtype=torch.float32)

    # Calculate uncompressed memory
    uncompressed_bytes = original_weights.element_size() * original_weights.nelement()
    print(f"Original FP32 Matrix Size: {uncompressed_bytes / (1024 ** 2):.4f} MB")

    # Step 1: Quantize weights to INT8
    quantized_weights, scale = quantize_int8(original_weights)
    compressed_bytes = quantized_weights.element_size() * quantized_weights.nelement()
    
    print(f"Quantized INT8 Matrix Size: {compressed_bytes / (1024 ** 2):.4f} MB")
    print(f"VRAM Memory Compression Ratio: {uncompressed_bytes / compressed_bytes:.2f}x Memory Reduction!")

    # Step 2: Dequantize back to FP32 to evaluate reconstruction quality
    reconstructed_weights = dequantize_int8(quantized_weights, scale)
    
    # Calculate Mean Absolute Error (MAE)
    mae_error = torch.mean(torch.abs(original_weights - reconstructed_weights)).item()
    print(f"Quantization Distortion Error (MAE): {mae_error:.6f}\n")


def ollama_quantized_demo():
    """
    Demonstrates GGUF quantized model execution using Ollama client.
    """
    print("=" * 70)
    print("2. Ollama GGUF Quantized Engine Execution")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        print(f"Executing inference on GGUF model '{model_name}'...")
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Explain 4-bit AWQ quantization in 2 bullet points."}]
        )
        t1 = time.perf_counter()
        print(f"Inference Latency: {t1 - t0:.2f} s")
        print(f"Response: {resp['message']['content']}\n")

    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    torch_quantization_demo()
    ollama_quantized_demo()
