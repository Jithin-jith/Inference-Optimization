"""
================================================================================
MODULE 16: GEMINI REAL-TIME RESPONSE STREAMING (`generate_content_stream`)
================================================================================

CONCEPT OVERVIEW:
-----------------
Google Gemini API provides native streaming support via `generate_content_stream`.

HOW IT WORKS:
-------------
- The SDK opens a HTTP/2 chunked transfer connection to Google's inference servers.
- As tokens are sampled from TPU softmax outputs, Google streams `GenerateContentResponse` event
  chunks back to the client immediately.
- Frontend applications render text token-by-token, bringing perceived user latency down to ~150ms.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Using `client.models.generate_content_stream(...)` in the modern `google-genai` SDK, tracking TTFT
and total stream duration.
================================================================================
"""

import os
import sys
import time
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types


def gemini_streaming_demo():
    """
    Executes real-time token response streaming with Google Gemini 2.5 Flash.
    """
    print("=" * 70)
    print("Google Gemini API Real-Time Response Streaming (`generate_content_stream`)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY environment variable is not set.")
        print("To run live streaming calls, set export GEMINI_API_KEY='your_api_key'.\n")

    client = genai.Client()
    model_name = "gemini-2.5-flash"

    prompt = "Write a concise paragraph detailing why token-by-token streaming improves web application user experience."

    try:
        print("Opening streaming connection to Google Gemini API...\nResponse Stream: ")
        t0 = time.perf_counter()
        first_token = True
        ttft_s = 0.0

        # Execute streaming API call
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200
            )
        )

        # Iterate over streaming chunks as they arrive from Google servers
        for chunk in response_stream:
            if first_token:
                ttft_s = time.perf_counter() - t0
                first_token = False
            sys.stdout.write(chunk.text)
            sys.stdout.flush()

        t1 = time.perf_counter()
        print("\n" + "-" * 70)
        print(f"Perceived Time to First Token (TTFT): {ttft_s:.3f} seconds")
        print(f"Total Response Generation Time:      {t1 - t0:.3f} seconds\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API streaming call skipped or failed ({e}).")


if __name__ == "__main__":
    gemini_streaming_demo()
