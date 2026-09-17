"""
================================================================================
MODULE 10: GEMINI DISTRIBUTED TPU TENSOR SHARDING
================================================================================

CONCEPT OVERVIEW:
-----------------
Google Gemini 2.5 Pro contains hundreds of billions of parameters. To serve low-latency responses,
Google's infrastructure shards model parameters across 2D/3D TPU Pod topologies (e.g. TPU v5e / v6e).

TPU INTER-CHIP INTERCONNECT (ICI):
----------------------------------
- Google TPUs connect via proprietary Inter-Chip Interconnect (ICI) providing 1.6 TB/s bi-directional bandwidth.
- JAX / XLA automatically generates Megatron-style Column and Row Tensor Parallel execution graphs,
  enabling sub-millisecond All-Reduce collectives across TPU cores.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using `google-genai` SDK to query Gemini Pro regarding TPU Tensor Sharding architectures.
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



def gemini_tensor_parallelism_demo():
    """
    Queries Google Gemini API regarding TPU Tensor Sharding architectures.
    """
    print("=" * 70)
    print("Google Gemini Infrastructure: Distributed TPU Tensor Sharding")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Warning] GOOGLE_API_KEY environment variable is not set.")
        print("To run live, set export GOOGLE_API_KEY='your_api_key'.\n")

    client = genai.Client()
    model_name = "gemini-3.1-pro-preview"

    prompt = "Explain how Google TPU v5e Megatron-style Tensor Parallelism shards multi-head attention across ICI chips."

    try:
        print(f"Sending tensor sharding architecture query to '{model_name}'...")
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=250
            )
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response Snippet:\n{response.text[:300]}...\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    gemini_tensor_parallelism_demo()
