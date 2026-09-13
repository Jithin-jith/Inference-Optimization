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
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To run live, set export GEMINI_API_KEY='your_api_key'.\n")

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
