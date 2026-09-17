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
import sys
import time
import warnings
import logging
from typing import TypedDict
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Filter out lower-level SDK warnings written directly to sys.stderr
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, msg):
        if "automatic function calling" in msg or "AFC" in msg:
            return
        self.original_stderr.write(msg)

    def flush(self):
        if hasattr(self.original_stderr, "flush"):
            self.original_stderr.flush()

sys.stderr = StderrFilter(sys.stderr)

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
warnings.showwarning = lambda *args, **kwargs: None

logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

load_dotenv()



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
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Warning] GOOGLE_API_KEY environment variable is not set.")
        print("To run live, set export GOOGLE_API_KEY='your_api_key'.\n")

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
