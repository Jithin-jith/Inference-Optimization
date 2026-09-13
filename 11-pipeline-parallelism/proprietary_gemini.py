"""
================================================================================
MODULE 11: GEMINI MULTI-CLUSTER PIPELINE ARCHITECTURE
================================================================================

CONCEPT OVERVIEW:
-----------------
Google dispatches giant multi-trillion parameter Gemini models across TPU Pod slices linked via
Optical Circuit Switches (OCS) using pipeline execution graphs.

MULTI-NODE NETWORKING:
----------------------
In proprietary cloud infrastructure, pipeline stages pass intermediate activation tensors
across high-speed optical network links while TPU clusters run micro-batch pipelines in parallel.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using `google-genai` SDK to query Gemini API regarding multi-cluster pipeline architecture.
================================================================================
"""

import os
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


def gemini_pipeline_architecture_demo():
    """
    Queries Google Gemini API regarding multi-node pipeline parallelism.
    """
    print("=" * 70)
    print("Google Gemini Infrastructure: Multi-Node Pipeline Parallelism")
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

    prompt = "Explain how Google dispatches giant 1-trillion parameter Gemini models across TPU Pods using Pipeline Parallelism."

    try:
        print("Querying Gemini Cloud API...")
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
    gemini_pipeline_architecture_demo()
