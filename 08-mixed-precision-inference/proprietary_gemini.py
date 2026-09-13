"""
================================================================================
MODULE 08: GEMINI BFLOAT16 & HARDWARE PRECISION ARCHITECTURE
================================================================================

CONCEPT OVERVIEW:
-----------------
Google originally invented **Bfloat16 (BF16)** specifically to optimize TPU neural network training
and inference. Unlike FP16 (which has 5 exponent bits and is prone to underflow/overflow), BF16 keeps
the exact same 8 exponent bits as FP32, providing identical dynamic range while cutting memory size in half!

GEMINI HARDWARE PRECISION PIPELINE:
-----------------------------------
- Google TPU v4 / v5e / v6e execute matrix multiplications natively in Bfloat16.
- High-QPS serverless endpoints utilize FP8 and INT8 matrix operations to maximize throughput.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Querying Google Gemini 2.5 Flash using the `google-genai` SDK.
================================================================================
"""

import os
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


def gemini_precision_analysis_demo():
    """
    Queries Google Gemini API regarding Bfloat16 hardware advantages.
    """
    print("=" * 70)
    print("Google Gemini API: Bfloat16 & TPU Hardware Precision Architecture")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To run live, set export GEMINI_API_KEY='your_api_key'.\n")

    client = genai.Client()
    model_name = "gemini-2.5-flash"

    prompt = "Explain why Bfloat16 (BF16) is preferred over FP16 for deep learning TPU/GPU inference."

    try:
        print(f"Sending precision analysis query to Gemini model '{model_name}'...")
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=200
            )
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response Snippet:\n{response.text[:250]}...\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    gemini_precision_analysis_demo()
