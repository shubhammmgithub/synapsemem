import statistics
import time

from synapsemem import SynapseMemory
from benchmarks.sample_data import SAMPLE_MEMORIES


def run_ingest_benchmark(rounds: int = 3):
    all_times = []

    for _ in range(rounds):
        memory = SynapseMemory()

        for text in SAMPLE_MEMORIES:
            start = time.perf_counter()
            memory.ingest(text)
            end = time.perf_counter()
            all_times.append(end - start)

    print("\n=== Ingest Benchmark ===")
    print(f"Samples: {len(SAMPLE_MEMORIES) * rounds}")
    print(f"Average ingest time: {statistics.mean(all_times):.6f} sec")
    print(f"Median ingest time:  {statistics.median(all_times):.6f} sec")
    print(f"Min ingest time:     {min(all_times):.6f} sec")
    print(f"Max ingest time:     {max(all_times):.6f} sec")


if __name__ == "__main__":
    run_ingest_benchmark()