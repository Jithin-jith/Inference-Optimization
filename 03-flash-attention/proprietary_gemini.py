"""
================================================================================
MODULE 03: GEMINI MILLION-TOKEN LONG-CONTEXT PROCESSING
================================================================================

CONCEPT OVERVIEW:
-----------------
Google Gemini models (such as `gemini-2.5-flash` and `gemini-2.5-pro`) support context windows
of 1,000,000 to 2,000,000+ tokens.

HOW GOOGLE ACHIEVES THIS AT THE INFRASTRUCTURE LAYER:
------------------------------------------------------
Under the hood, Google Cloud TPU clusters compile attention graphs using:
1. TPU XLA Block-Tiling (similar to FlashAttention SRAM tiling).
2. Ring-Attention: Splitting the 1M+ token sequence across multiple TPU Pod chips connected
   via high-speed Inter-Chip Interconnect (ICI) operating at 1.6 TB/s.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using the official `google-genai` SDK to execute a Needle-in-a-Haystack search over a large document dataset
(25,000+ tokens) with zero context degradation.
================================================================================
"""

import os
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


def gemini_long_context_demo():
    """
    Executes a long-context document query using Google Gemini 2.5 Flash.
    """
    print("=" * 70)
    print("Google Gemini API: 1 Million+ Token Long-Context Architecture")
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

    # -------------------------------------------------------------------------
    # Step 1: Construct Synthetic Large Document (~25,000 tokens)
    # -------------------------------------------------------------------------
    print("1. Constructing large context log dataset (~2,500 lines)...")
    document_lines = [f"Line {i}: Customer log entry - System status OK, metric value = {i * 13 % 100}." for i in range(2500)]
    
    # Plant a "Needle in a Haystack" target information at line 1235
    document_lines.insert(1234, "CRITICAL ERROR: Line 1235 contains system security token 'X99-SECRET-KEY-999'.")
    full_document = "\n".join(document_lines)

    print(f" -> Total Context Size: ~{len(full_document.split())} words (~25,000 tokens).")

    query = "Find the exact line number and secret key value mentioned in the system log document."

    try:
        # ---------------------------------------------------------------------
        # Step 2: Query Gemini Long-Context Endpoint
        # ---------------------------------------------------------------------
        print("\n2. Executing Needle-in-a-Haystack search over Long Context...")
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_name,
            contents=[full_document, query],
            config=types.GenerateContentConfig(
                temperature=0.0  # Zero temperature for exact factual extraction
            )
        )
        t1 = time.perf_counter()
        
        print(f" -> Processing Latency: {t1 - t0:.2f} s")
        print(f" -> Token Usage Metadata: {response.usage_metadata}")
        print(f" -> Model Answer:\n{response.text}\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    gemini_long_context_demo()
