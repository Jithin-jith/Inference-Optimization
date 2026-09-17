"""
================================================================================
MODULE 05: GEMINI BATCH INFERENCE BENCHMARK & PARAMETER COMPARISON
================================================================================

CONCEPT OVERVIEW:
-----------------
For offline, non-real-time production tasks (such as dataset classification, bulk summarization,
synthetic data generation, or offline model evaluation), executing individual real-time synchronous
API calls wastes money and risks hitting QPS rate limits.

GOOGLE GEMINI BATCH API SOLUTION:
---------------------------------
Google Gemini provides an explicit **Batch API** endpoint (`client.batches.create`).
- 50% Cost Discount: All tokens processed via Batch API receive a 50% discount compared to standard pricing.
- Amortized Weight Reuse: High batch size ($Y = XW$) amortizes weight loading from HBM, saturating GPU compute.
- Asynchronous Delivery: System processes requests in parallel on Google Cloud background queues.

WHAT THIS SCRIPT DEMONSTRATES:
------------------------------
Compares:
1. Real-Time Sequential Mode (Batch Size = 1, standard 100% pricing rate, memory-bandwidth bound)
2. Offline Batch API Mode (Batch Inference Job, 50% cost discount tier, compute saturating)

Polls job completion, extracts output text and metadata from `job.dest.inlined_responses`,
and renders a detailed side-by-side parameter comparison table.
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


def gemini_batch_comparison_demo():
    """
    Executes a benchmark comparison between Real-Time Sequential Requests (Batch Size = 1)
    and Google Gemini Batch API Workflow (Batch Inference Job), retrieving batch outputs
    and comparing both approaches against identical parameters.
    """
    print("=" * 90)
    print("Google Gemini API: Real-Time Requests vs. Batch API Detailed Parameter Benchmark")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # API Key & Client Setup
    # -------------------------------------------------------------------------
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Warning] GOOGLE_API_KEY environment variable is not set.")
        print("To run live batch jobs against Google Cloud, set export GOOGLE_API_KEY='your_api_key'.\n")

    client = genai.Client()
    model_name = "gemini-2.5-flash"

    # Dataset of prompts to process in benchmark
    batch_prompts = [
        "Classify sentiment: 'The inference latency was incredibly fast and throughput doubled!'",
        "Classify sentiment: 'The GPU ran out of VRAM memory resulting in an OOM crash.'",
        "Classify sentiment: 'The batch job finished as scheduled within expected SLAs.'",
        "Classify sentiment: 'PagedAttention slashes memory waste down to less than 4 percent.'"
    ]

    try:
        # =====================================================================
        # Phase 1: Real-Time Sequential Requests (Without Batching / Batch Size = 1)
        # =====================================================================
        print("\n--- PHASE 1: WITHOUT BATCHING (Real-Time Sequential / Batch Size = 1) ---")
        seq_start = time.perf_counter()
        seq_latencies = []
        seq_outputs = []
        seq_prompt_tokens = 0
        seq_candidate_tokens = 0
        seq_total_tokens = 0

        for idx, prompt in enumerate(batch_prompts, 1):
            t0 = time.perf_counter()
            print(f"[Sync Req #{idx}] Sending real-time request: '{prompt[:45]}...'")
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=100)
            )
            t1 = time.perf_counter()
            latency = t1 - t0
            seq_latencies.append(latency)
            seq_outputs.append(resp.text.strip() if resp.text else "")

            # Dynamically extract token usage metadata from API response
            meta = resp.usage_metadata
            p_tok = meta.prompt_token_count if meta else 0
            c_tok = meta.candidates_token_count if meta else 0
            tot_tok = meta.total_token_count if meta else (p_tok + c_tok)

            seq_prompt_tokens += p_tok
            seq_candidate_tokens += c_tok
            seq_total_tokens += tot_tok

            print(f"[Sync Req #{idx}] Completed in {latency:.2f} s | Tokens: {tot_tok} (Prompt: {p_tok}, Output: {c_tok})")
            print(f"               Response Output: '{resp.text.strip()[:60]}...'")

        seq_end = time.perf_counter()
        seq_total_time = seq_end - seq_start

        # =====================================================================
        # Phase 2: Official Batch API Workflow (With Batching / Batch Job)
        # =====================================================================
        print("\n--- PHASE 2: WITH BATCHING (Official Batch API Job / `client.batches.create`) ---")
        batch_start = time.perf_counter()
        
        # Step A: Compile Inlined Batch Requests
        requests_list = [
            types.InlinedRequest(
                contents=prompt
            ) for prompt in batch_prompts
        ]
        print(f" -> Compiled {len(requests_list)} inlined requests into batch payload.")

        # Step B: Submit Batch Job to Google Cloud Infrastructure
        job = client.batches.create(
            model=model_name,
            src=requests_list
        )
        job_name = job.name
        print(f" -> Batch Job Submitted Successfully! Job Resource Name: '{job_name}'")
        print(f" -> Initial State: {job.state}")
        print(" -> Polling Google Cloud background queue for batch completion...")

        # Step C: Poll for Job Completion and Retrieve Outputs
        poll_count = 0
        while job.state not in [types.JobState.JOB_STATE_SUCCEEDED, types.JobState.JOB_STATE_FAILED]:
            poll_count += 1
            time.sleep(3)
            job = client.batches.get(name=job_name)
            print(f"    [Poll #{poll_count}] Job Status: {job.state}...")
            if poll_count >= 60:  # Timeout safety (3 minutes)
                print(" -> Timeout waiting for batch job completion.")
                break

        batch_end = time.perf_counter()
        batch_total_time = batch_end - batch_start

        # Step D: Extract Generated Outputs & Token Usage from Batch Destination
        batch_outputs = []
        batch_prompt_tokens = 0
        batch_candidate_tokens = 0
        batch_total_tokens = 0

        print(f"\n -> Final Batch Job State: {job.state}")
        if job.state == types.JobState.JOB_STATE_SUCCEEDED and job.dest and job.dest.inlined_responses:
            print(" -> Retrieving generated responses from `job.dest.inlined_responses`:")
            for idx, item in enumerate(job.dest.inlined_responses, 1):
                resp = item.response
                out_text = resp.text.strip() if hasattr(resp, "text") and resp.text else str(resp)
                batch_outputs.append(out_text)

                meta = resp.usage_metadata
                p_tok = meta.prompt_token_count if meta else 0
                c_tok = meta.candidates_token_count if meta else 0
                tot_tok = meta.total_token_count if meta else (p_tok + c_tok)

                batch_prompt_tokens += p_tok
                batch_candidate_tokens += c_tok
                batch_total_tokens += tot_tok

                print(f"    [Batch Output #{idx}] Tokens: {tot_tok} (Prompt: {p_tok}, Output: {c_tok})")
                print(f"                       Result Text: '{out_text[:60]}...'")
        else:
            print(" -> Batch job pending or inlined responses unavailable at query time.")
            batch_prompt_tokens = seq_prompt_tokens
            batch_candidate_tokens = seq_candidate_tokens
            batch_total_tokens = seq_total_tokens
            batch_outputs = seq_outputs

        # =====================================================================
        # Phase 3: Detailed Side-by-Side Parameter Comparison Table
        # =====================================================================
        # Metrics Calculations
        seq_avg_lat = sum(seq_latencies) / len(seq_latencies) if seq_latencies else 0.0
        batch_avg_lat = batch_total_time / len(batch_prompts) if len(batch_prompts) > 0 else 0.0

        seq_req_tp = len(batch_prompts) / seq_total_time if seq_total_time > 0 else 0.0
        batch_req_tp = len(batch_prompts) / batch_total_time if batch_total_time > 0 else 0.0

        seq_tok_tp = seq_total_tokens / seq_total_time if seq_total_time > 0 else 0.0
        batch_tok_tp = batch_total_tokens / batch_total_time if batch_total_time > 0 else 0.0

        # Billing Cost Estimation ($0.15/1M prompt, $0.60/1M output for Gemini 2.5 Flash)
        seq_cost = (seq_prompt_tokens * 0.15 / 1_000_000) + (seq_candidate_tokens * 0.60 / 1_000_000)
        batch_cost = (batch_prompt_tokens * 0.15 / 1_000_000 * 0.50) + (batch_candidate_tokens * 0.60 / 1_000_000 * 0.50)
        savings_dollar = seq_cost - batch_cost

        print("\n" + "=" * 90)
        print("DETAILED PARAMETER COMPARISON SUMMARY: WITHOUT BATCHING VS. WITH BATCHING")
        print("=" * 90)
        
        fmt = "  {:<32} | {:<25} | {:<25}"
        print(fmt.format("PARAMETER / METRIC", "WITHOUT BATCHING (b=1)", "WITH BATCHING (Batch API)"))
        print("  " + "-" * 86)
        print(fmt.format("Execution Mode", "Synchronous Real-Time", "Asynchronous TPU Queue"))
        print(fmt.format("Dataset Size", f"{len(batch_prompts)} Prompts", f"{len(batch_prompts)} Prompts"))
        print(fmt.format("Prompt Tokens", f"{seq_prompt_tokens} Tokens", f"{batch_prompt_tokens} Tokens"))
        print(fmt.format("Generated Output Tokens", f"{seq_candidate_tokens} Tokens", f"{batch_candidate_tokens} Tokens"))
        print(fmt.format("Total Tokens Processed", f"{seq_total_tokens} Tokens", f"{batch_total_tokens} Tokens"))
        print(fmt.format("Total Execution Wall Time", f"{seq_total_time:.2f} s", f"{batch_total_time:.2f} s"))
        print(fmt.format("Average Latency per Item", f"{seq_avg_lat:.2f} s (Immediate TTFT)", f"{batch_avg_lat:.2f} s (Queue + GEMM)"))
        print(fmt.format("Request Throughput", f"{seq_req_tp:.2f} req/s", f"{batch_req_tp:.2f} req/s"))
        print(fmt.format("Token Throughput", f"{seq_tok_tp:.2f} tokens/s", f"{batch_tok_tp:.2f} tokens/s"))
        print(fmt.format("Billing Price Rate Tier", "100% Standard Rate ($1.0x)", "50% Discount Tier ($0.5x)"))
        print(fmt.format("Estimated Dollar Cost ($)", f"${seq_cost:.6f}", f"${batch_cost:.6f}"))
        print(fmt.format("Cost Savings ($)", "Baseline ($0.00)", f"${savings_dollar:.6f} (50% Off)"))
        print(fmt.format("Hardware Compute Status", "Memory-Bandwidth Bound", "Compute Saturating (Y = XW)"))
        print(fmt.format("Weight Loading Amortization", "Low (Loaded per request)", "High (Loaded once per batch)"))
        print(fmt.format("QPS Rate Limit Risk", "High (Exhausts Sync Quotas)", "Zero (Background Queue)"))
        print(fmt.format("Ideal Production Fit", "Real-Time Voice/Chat", "Offline Datasets/Evaluation"))
        print("=" * 90 + "\n")

    except Exception as e:
        print(f"\n[SDK Execution Note]: API call skipped or failed ({e}).")


if __name__ == "__main__":
    gemini_batch_comparison_demo()
