"""
================================================================================
MODULE 06: GEMINI DYNAMIC ADAPTIVE ROUTER (FLASH VS PRO CASCADE)
================================================================================

CONCEPT OVERVIEW:
-----------------
Enterprise AI systems handle diverse workloads: simple factual questions vs complex software design.
Always dispatching prompts to heavy models (like `gemini-2.5-pro`) wastes budget and increases latency.

DYNAMIC CASCADE ROUTER PATTERN:
-------------------------------
1. Fast Intent Classification: Pass user query to `gemini-2.5-flash` with a concise prompt to classify intent:
   - SIMPLE: Route to `gemini-2.5-flash` (10x cheaper, fast response).
   - COMPLEX: Route to `gemini-2.5-pro` (Deep reasoning, multi-step math/code).

BENEFITS:
---------
- Slashes average system latency by 40% - 60%.
- Maximizes cost-efficiency across production workloads.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using the `google-genai` SDK to build a two-stage classifier-router pipeline.
================================================================================
"""

import os
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


def classify_query_intent(client, prompt: str) -> str:
    """
    Uses Gemini Flash to rapidly classify prompt complexity into SIMPLE or COMPLEX.
    """
    classification_prompt = f"""
    Classify the following query into exactly one category: 'SIMPLE' or 'COMPLEX'.
    - SIMPLE: Basic facts, short definitions, simple translations.
    - COMPLEX: Code generation, architectural design, complex math proofs, multi-step logic.

    Query: "{prompt}"
    Category:"""
    
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=classification_prompt,
            config=types.GenerateContentConfig(max_output_tokens=5, temperature=0.0)
        )
        return res.text.strip().upper()
    except Exception:
        return "COMPLEX"  # Fallback to heavy model on classifier failure


def gemini_adaptive_cascade_demo():
    """
    Demonstrates dynamic intent classification and model routing with Gemini API.
    """
    print("=" * 70)
    print("Google Gemini API: Adaptive Dynamic Model Cascade Router")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To run live, set export GEMINI_API_KEY='your_api_key'.\n")

    client = genai.Client()

    test_queries = [
        "What is the freezing point of water in Celsius?",
        "Design a high-throughput event-driven microservices architecture handling 1,000,000 requests/sec with Kafka and Go."
    ]

    for q in test_queries:
        print(f"\nIncoming User Prompt: '{q}'")
        t0 = time.perf_counter()
        
        # Step 1: Classify intent using fast Flash model
        intent = classify_query_intent(client, q)
        
        # Step 2: Select target model based on intent
        selected_model = "gemini-2.5-flash" if "SIMPLE" in intent else "gemini-2.5-pro"
        print(f" -> Classifier Intent: {intent} | Routed Target: {selected_model}")

        # Step 3: Execute prompt on target model
        try:
            resp = client.models.generate_content(
                model=selected_model,
                contents=q,
                config=types.GenerateContentConfig(max_output_tokens=150)
            )
            t1 = time.perf_counter()
            print(f" -> Total Execution Time: {t1 - t0:.2f} s")
            print(f" -> Response Snippet: {resp.text[:120]}...")
        except Exception as e:
            print(f" -> Execution Note: ({e})")


if __name__ == "__main__":
    gemini_adaptive_cascade_demo()
