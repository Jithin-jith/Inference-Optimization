"""
================================================================================
MODULE 09: GEMINI NANO INT4 EDGE & TPU INT8 QUANTIZATION
================================================================================

CONCEPT OVERVIEW:
-----------------
Quantization is used extensively across Google's model spectrum:
1. Gemini Nano (Mobile & Edge): Uses 4-bit INT4 quantization to run locally on Google Pixel smartphones
   within strict mobile thermal and battery envelopes.
2. Gemini Cloud (TPUs): Uses INT8 and FP8 matrix operations on Google TPU v5e/v6e to maximize QPS throughput.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Querying Google Gemini API using `google-genai` SDK to analyze edge INT4 vs cloud INT8 architectures.
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



def gemini_quantization_analysis_demo():
    """
    Queries Gemini API regarding mobile INT4 edge quantization.
    """
    print("=" * 70)
    print("Google Gemini Architecture: Mobile INT4 Edge & Cloud Quantization")
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

    prompt = "Explain how Gemini Nano achieves 4-bit INT4 quantization for real-time execution on mobile devices."

    try:
        print("Sending quantization architecture query to Gemini Cloud API...")
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
    gemini_quantization_analysis_demo()
