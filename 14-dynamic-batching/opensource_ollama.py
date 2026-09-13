"""
================================================================================
MODULE 14: CONTINUOUS BATCHING & ITERATION-LEVEL SCHEDULING
================================================================================

CONCEPT OVERVIEW:
-----------------
Traditional **Static Batching** groups requests at the request level.
If Request 1 generates 20 tokens and Request 2 generates 300 tokens:
- Request 1 completes early but is forced to wait in GPU memory until Request 2 finishes.
- Wastes up to 80% of GPU compute slots with empty PAD tokens.

CONTINUOUS BATCHING SOLUTION (ORCA / vLLM SCHEDULER):
------------------------------------------------------
Operates at the **iteration level** (single token step):
- As soon as Request 1 emits an `<EOS>` token, its slot is freed INSTANTLY.
- A new waiting request from the queue is inserted mid-generation into the running batch step!
- Boosts GPU utilization from ~15% to >85%!

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `ContinuousBatchScheduler`: Python class implementing an iteration-level dynamic batch insertion & ejection engine.
2. `run_dynamic_batch_simulation()`: Executes multi-request iteration queue simulation.
3. `ollama_dynamic_demo()`: Ollama multi-request execution overview.
================================================================================
"""

import time
import queue
# pyrefly: ignore [missing-import]
import ollama


class ContinuousBatchScheduler:
    """
    Simulates iteration-level continuous batch scheduling (Orca / vLLM architecture).
    Slots are freed and filled dynamically per token step.
    """
    def __init__(self, max_batch_slots=2):
        """
        Initialize batch scheduler with maximum active execution slots.
        """
        self.max_slots = max_batch_slots
        self.active_batch = {}  # Maps req_id -> remaining_tokens_to_generate
        self.wait_queue = queue.Queue()
        print(f"[Scheduler Initialized]: Max Active GPU Execution Slots = {max_batch_slots}")

    def add_request(self, req_id: str, total_tokens_to_gen: int):
        """Adds incoming user request to wait queue."""
        self.wait_queue.put((req_id, total_tokens_to_gen))
        print(f"Enqueued Request '{req_id}' (Total tokens to generate: {total_tokens_to_gen})")

    def run_iterations(self):
        """
        Executes iteration-level scheduling loop.
        """
        print("\nStarting Continuous Iteration Engine Loop...")
        step = 0
        while not self.wait_queue.empty() or self.active_batch:
            step += 1
            
            # --- Slot Insertion Check: Fill empty slots from wait queue ---
            while len(self.active_batch) < self.max_slots and not self.wait_queue.empty():
                req_id, tokens = self.wait_queue.get()
                self.active_batch[req_id] = tokens
                print(f" -> Iteration #{step}: [MID-FLIGHT INSERTION] Inserted '{req_id}' into active slot!")

            # --- Decode Pass: Execute 1 token generation step for active slots ---
            finished_requests = []
            for req_id in list(self.active_batch.keys()):
                self.active_batch[req_id] -= 1
                if self.active_batch[req_id] == 0:
                    finished_requests.append(req_id)

            # --- Slot Ejection Check: Immediately free slots on completion ---
            for req_id in finished_requests:
                del self.active_batch[req_id]
                print(f" -> Iteration #{step}: [EJECTION] '{req_id}' completed! Slot freed immediately.")

        print("All Queued Requests Processed Successfully!\n")


def run_dynamic_batch_simulation():
    """
    Executes continuous batching simulation with requests of varying lengths.
    """
    print("=" * 70)
    print("1. Continuous Iteration-Level Batching Scheduler Simulation")
    print("=" * 70)

    scheduler = ContinuousBatchScheduler(max_batch_slots=2)

    # Enqueue requests with varying generation token lengths
    scheduler.add_request("Req_A (Short - 3 tokens)", 3)
    scheduler.add_request("Req_B (Long  - 6 tokens)", 6)
    scheduler.add_request("Req_C (Medium- 4 tokens)", 4)

    scheduler.run_iterations()


def ollama_dynamic_demo():
    """
    Demonstrates Ollama multi-request execution overview.
    """
    print("=" * 70)
    print("2. Ollama Dynamic Multi-Request Execution")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Explain continuous batching vs static batching in 2 sentences."}]
        )
        t1 = time.perf_counter()
        print(f"Execution Latency: {t1 - t0:.2f} s")
        print(f"Response: {resp['message']['content']}\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    run_dynamic_batch_simulation()
    ollama_dynamic_demo()
