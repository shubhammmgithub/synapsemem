import time

from synapsemem import SynapseMemory
from benchmarks.sample_data import SLEEP_TEST_MEMORIES, SLEEP_TEST_QUERY


def _count_by(records, key, value):
    return sum(1 for r in records if r.get(key) == value)


def run_sleep_benchmark():
    memory = SynapseMemory(storage_backend="memory")
    memory.sleep_consolidator = memory.sleep_consolidator.__class__(
        min_age_seconds=1,
        prune_age_seconds=1,
        promotion_min_support=2,
        promotion_min_priority=4,
    )

    for text in SLEEP_TEST_MEMORIES:
        memory.ingest(text)

    print("\n=== Sleep Consolidation Benchmark ===")

    active_before = memory.storage.all()
    all_before = memory.storage.all_records() if hasattr(memory.storage, "all_records") else active_before
    retrieve_before = memory.retrieve(SLEEP_TEST_QUERY, top_k=5)

    print("\nBefore sleep")
    print(f"Active records: {len(active_before)}")
    print(f"Total records:  {len(all_before)}")
    print(f"Episodic active: {_count_by(active_before, 'memory_type', 'episodic')}")
    print(f"Semantic active: {_count_by(active_before, 'memory_type', 'semantic')}")

    if retrieve_before:
        top = retrieve_before[0]
        print("Top retrieval before sleep:")
        print(
            f"  {top['subject']} {top['predicate']} {top['object']} "
            f"[type={top.get('memory_type')}, score={top.get('score')}]"
        )

    time.sleep(2)
    sleep_report = memory.sleep_consolidate(dry_run=False)

    active_after = memory.storage.all()
    all_after = memory.storage.all_records() if hasattr(memory.storage, "all_records") else active_after
    retrieve_after = memory.retrieve(SLEEP_TEST_QUERY, top_k=5)

    print("\nSleep report")
    print(f"Promoted: {sleep_report.get('promoted', 0)}")
    print(f"Merged:   {sleep_report.get('merged', 0)}")
    print(f"Pruned:   {sleep_report.get('pruned', 0)}")
    print(f"Kept:     {sleep_report.get('kept', 0)}")

    print("\nAfter sleep")
    print(f"Active records: {len(active_after)}")
    print(f"Total records:  {len(all_after)}")
    print(f"Episodic active: {_count_by(active_after, 'memory_type', 'episodic')}")
    print(f"Semantic active: {_count_by(active_after, 'memory_type', 'semantic')}")
    print(f"Merged total: {_count_by(all_after, 'status', 'merged')}")
    print(f"Pruned total: {_count_by(all_after, 'status', 'pruned')}")

    if retrieve_after:
        top = retrieve_after[0]
        print("Top retrieval after sleep:")
        print(
            f"  {top['subject']} {top['predicate']} {top['object']} "
            f"[type={top.get('memory_type')}, score={top.get('score')}, "
            f"semantic_bonus={top.get('semantic_memory_bonus')}, "
            f"source_count_bonus={top.get('source_count_bonus')}]"
        )


if __name__ == "__main__":
    run_sleep_benchmark()