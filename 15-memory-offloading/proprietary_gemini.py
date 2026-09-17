"""
================================================================================
MODULE 15: GEMINI MANAGED SERVERLESS MEMORY ABSTRACTION
================================================================================

CONCEPT OVERVIEW:
-----------------
In serverless cloud APIs (like Google Gemini), physical GPU/TPU memory boundaries are completely
abstracted away from the developer.

SERVERLESS ABSTRACTION ADVANTAGES:
----------------------------------
- No manual GPU VRAM offloading configuration required.
- No PCIe bus bottleneck management or Out-Of-Memory (OOM) crashes.
- Google infrastructure dynamically provisions multi-node TPU clusters per request.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using `google-genai` SDK to query Gemini Flash regarding serverless memory abstraction.
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



def gemini_memory_offloading_demo():
    """
    Queries Google Gemini API regarding serverless memory abstraction.
    """
    print("=" * 70)
    print("Google Gemini API: Managed Serverless Memory Abstraction")
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

    prompt = "Explain why serverless APIs like Gemini remove the operational need for manual GPU VRAM offloading."

    try:
        print("Sending memory abstraction query to Gemini Cloud API...")
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
    gemini_memory_offloading_demo()
