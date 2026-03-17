from benchmarks.benchmark_ingest import run_ingest_benchmark
from benchmarks.benchmark_retrieve import run_retrieve_benchmark
from benchmarks.benchmark_prompt import run_prompt_benchmark
from benchmarks.benchmark_quality import run_quality_benchmark


if __name__ == "__main__":
    run_ingest_benchmark()
    run_retrieve_benchmark()
    run_prompt_benchmark()
    run_quality_benchmark()