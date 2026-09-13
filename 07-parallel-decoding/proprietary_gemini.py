"""
================================================================================
MODULE 07: GEMINI PARALLEL STRUCTURED OUTPUT GENERATION
================================================================================

CONCEPT OVERVIEW:
-----------------
Generating structured outputs (like strict JSON adhering to Pydantic schemas) requires models to generate
syntax tokens (brackets, key names, commas) alongside value tokens.

PARALLEL MULTI-TOKEN GENERATION IN GEMINI:
------------------------------------------
Google Gemini leverages internal multi-token heads and constrained grammar decoding to emit
structured JSON payloads at high speed.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using the `google-genai` SDK with `response_mime_type="application/json"` and `response_schema`
(defined via Pydantic BaseModel) to enforce strict, high-speed multi-token structured responses.
================================================================================
"""

import os
import time
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


# Define Pydantic Schema for Structured Parallel JSON Output
class OptimizationBenchmark(BaseModel):
    technique_name: str = Field(description="Name of the inference optimization technique")
    primary_metric: str = Field(description="Target metric improved e.g. Latency, Throughput, VRAM")
    speedup_factor: str = Field(description="Typical performance speedup multiplier e.g. 2.5x")


def gemini_parallel_structured_demo():
    """
    Demonstrates Gemini SDK structured JSON schema generation.
    """
    print("=" * 70)
    print("Google Gemini API: Parallel Structured Output Generation")
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

    prompt = "Provide structured benchmark metrics for FlashAttention, KV-Caching, and Speculative Decoding."

    try:
        print("Executing Structured Multi-Token JSON Generation with Gemini...")
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[OptimizationBenchmark],  # Enforces Pydantic Schema!
                temperature=0.1,
            )
        )
        t1 = time.perf_counter()
        print(f"Generation Latency: {t1 - t0:.2f} s")
        print("Structured Parallel Result:\n", response.text)

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    gemini_parallel_structured_demo()
