"""
================================================================================
MODULE 04: PAGEDATTENTION MEMORY MANAGER SIMULATION
================================================================================

CONCEPT OVERVIEW:
-----------------
In standard LLM serving frameworks, KV-cache memory for a request is allocated as a single,
contiguous block of GPU VRAM based on the MAXIMUM potential context length (e.g. 2,048 tokens).

PROBLEM WITH STANDARD ALLOCATION:
---------------------------------
1. Internal Fragmentation: Memory reserved for max sequence length that is never generated.
2. External Fragmentation: Gaps between contiguous sequence allocations.
- Over 60% - 80% of GPU VRAM is wasted as pre-allocated, unused padding!

PAGEDATTENTION SOLUTION (VIRTUAL MEMORY PAGING):
------------------------------------------------
Pioneered by vLLM:
- KV-caches are divided into fixed-size **Physical KV Blocks** (e.g., 16 tokens per block).
- Memory is allocated **dynamically block-by-block** as new tokens are generated.
- A **Block Table** maps logical sequence token indices to non-contiguous physical VRAM page addresses.
- Slashes VRAM waste from ~70% down to <4%, boosting server concurrency by 2x - 4x!

WHAT THIS SCRIPT CONTAINS:
--------------------------
1. `PagedAttentionMemoryManager`: Structural Python implementation of a Virtual Memory Page Allocator,
   managing physical block pools, request block tables, and dynamic allocation/deallocation.
2. `ollama_concurrent_demo()`: Execution overview of session management.
================================================================================
"""

import time
import math
# pyrefly: ignore [missing-import]
import ollama


class PagedAttentionMemoryManager:
    """
    Simulates vLLM OS-style Virtual Memory Block Allocator for KV-caches.
    Maps logical request token blocks to physical GPU memory pages.
    """
    
    def __init__(self, total_gpu_blocks: int, block_size: int = 16):
        """
        Initialize physical GPU memory page pool.
        :param total_gpu_blocks: Total number of physical memory pages in GPU VRAM pool.
        :param block_size: Number of tokens stored per physical block (default: 16).
        """
        self.total_blocks = total_gpu_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_gpu_blocks))  # List of available physical block IDs
        self.block_tables = {}                            # Maps request_id -> list of allocated physical block IDs
        
        print(f"[PagedAttention Allocator Initialized]: {total_gpu_blocks} Physical Pages (Block Size: {block_size} tokens)")
        print(f" Total Physical VRAM Capacity: {total_gpu_blocks * block_size} tokens\n")

    def allocate_request(self, request_id: str, prompt_token_count: int):
        """
        Allocates exact minimum physical blocks required for initial prompt tokens.
        """
        # Calculate minimum blocks required: ceil(prompt_tokens / block_size)
        num_blocks_needed = math.ceil(prompt_token_count / self.block_size)
        
        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError(f"OOM: Insufficient physical blocks available! Needed: {num_blocks_needed}, Free: {len(self.free_blocks)}")

        allocated_blocks = []
        for _ in range(num_blocks_needed):
            allocated_blocks.append(self.free_blocks.pop(0))
        
        self.block_tables[request_id] = allocated_blocks
        print(f"Allocated Request '{request_id}' ({prompt_token_count} tokens):")
        print(f" -> Required Blocks: {num_blocks_needed} | Assigned Physical Page IDs: {allocated_blocks}")

    def append_token(self, request_id: str, current_token_count: int):
        """
        Dynamically allocates a new physical block ONLY when current block reaches capacity.
        """
        # Check if the new token crosses block boundary (e.g. token 17, 33, 49...)
        if current_token_count % self.block_size == 1:
            if not self.free_blocks:
                raise MemoryError(f"OOM: GPU Page Pool exhausted when growing {request_id}!")
            new_block_id = self.free_blocks.pop(0)
            self.block_tables[request_id].append(new_block_id)
            print(f" -> [DYNAMIC ALLOCATION]: Request '{request_id}' crossed boundary at Token #{current_token_count}!")
            print(f"    Assigned NEW Physical Page ID {new_block_id}. Page Table: {self.block_tables[request_id]}")

    def free_request(self, request_id: str):
        """
        Frees allocated physical blocks back to the pool when request completes generation.
        """
        if request_id in self.block_tables:
            freed_blocks = self.block_tables.pop(request_id)
            self.free_blocks.extend(freed_blocks)
            print(f"Freed Request '{request_id}':")
            print(f" -> Returned {len(freed_blocks)} physical pages to pool. Total Free Pool: {len(self.free_blocks)} blocks")


def run_paged_attention_simulation():
    """
    Simulates dynamic memory page allocation across multiple concurrent user requests.
    """
    print("=" * 70)
    print("1. Virtual Memory PagedAttention Allocator Simulation")
    print("=" * 70)

    # Initialize manager with 10 physical blocks (total capacity 160 tokens)
    manager = PagedAttentionMemoryManager(total_gpu_blocks=10, block_size=16)

    # 1. Allocate Request A (User Chat: 35 tokens -> needs 3 blocks)
    manager.allocate_request("Req_A (User Chat)", prompt_token_count=35)
    
    # 2. Allocate Request B (Batch Scoring: 48 tokens -> needs 3 blocks)
    manager.allocate_request("Req_B (Batch Task)", prompt_token_count=48)

    # 3. Simulate generation for Request A: growing from 35 to 49 tokens
    print("\nSimulating autoregressive decoding for Req_A...")
    for t in range(36, 50):
        manager.append_token("Req_A (User Chat)", current_token_count=t)

    # 4. Request B finishes generation -> free pages back to physical pool
    print("\nReq_B finished generation.")
    manager.free_request("Req_B (Batch Task)")
    print("-" * 70 + "\n")


def ollama_concurrent_demo():
    """
    Demonstrates session management execution with Ollama.
    """
    print("=" * 70)
    print("2. Ollama Multi-Session Execution")
    print("=" * 70)

    model_name = "llama3.2:1b"
    try:
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Explain virtual memory paging in 2 bullet points."}]
        )
        t1 = time.perf_counter()
        print(f"Response Time: {t1 - t0:.2f} s")
        print(f"Response: {resp['message']['content']}\n")
    except Exception as e:
        print(f"[Ollama Notice]: Live call skipped ({e}).")


if __name__ == "__main__":
    run_paged_attention_simulation()
    ollama_concurrent_demo()
