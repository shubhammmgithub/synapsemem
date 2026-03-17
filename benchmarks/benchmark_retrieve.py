import statistics
import time

from synapsemem import SynapseMemory
from benchmarks.sample_data import SAMPLE_MEMORIES, SAMPLE_QUERIES


def run_retrieve_benchmark(rounds: int = 3, top_k: int = 5):
    all_times = []

    for _ in range(rounds):
        memory = SynapseMemory()

        for text in SAMPLE_MEMORIES:
            memory.ingest(text)

        for query in SAMPLE_QUERIES:
            start = time.perf_counter()
            _ = memory.retrieve(query, top_k=top_k)
            end = time.perf_counter()
            all_times.append(end - start)

    print("\n=== Retrieval Benchmark ===")
    print(f"Queries: {len(SAMPLE_QUERIES) * rounds}")
    print(f"Average retrieve time: {statistics.mean(all_times):.6f} sec")
    print(f"Median retrieve time:  {statistics.median(all_times):.6f} sec")
    print(f"Min retrieve time:     {min(all_times):.6f} sec")
    print(f"Max retrieve time:     {max(all_times):.6f} sec")


if __name__ == "__main__":
    run_retrieve_benchmark()