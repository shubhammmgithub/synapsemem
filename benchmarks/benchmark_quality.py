import time

from synapsemem import SynapseMemory
from benchmarks.sample_data import QUALITY_TEST_CASES


def _best_objects(results):
    return " | ".join(result["object"] for result in results)


def run_quality_benchmark():
    passed_before = 0
    passed_after = 0
    total = len(QUALITY_TEST_CASES)

    print("\n=== Retrieval Quality Benchmark ===")

    for i, case in enumerate(QUALITY_TEST_CASES, start=1):
        memory = SynapseMemory(storage_backend="memory")
        memory.sleep_consolidator = memory.sleep_consolidator.__class__(
            min_age_seconds=1,
            prune_age_seconds=1000,
            promotion_min_support=2,
            promotion_min_priority=4,
        )

        # ingest twice to enable semantic promotion where relevant
        memory.ingest(case["memory"])
        memory.ingest(case["memory"])

        results_before = memory.retrieve(case["query"], top_k=3)
        objects_before = _best_objects(results_before)
        success_before = case["expected_substring"].lower() in objects_before.lower()

        if success_before:
            passed_before += 1

        time.sleep(2)
        sleep_report = memory.sleep_consolidate(dry_run=False)

        results_after = memory.retrieve(case["query"], top_k=3)
        objects_after = _best_objects(results_after)
        success_after = case["expected_substring"].lower() in objects_after.lower()

        if success_after:
            passed_after += 1

        top_type_after = results_after[0].get("memory_type") if results_after else None

        print(f"\nTest {i}")
        print(f"Memory:   {case['memory']}")
        print(f"Query:    {case['query']}")
        print(f"Expected: {case['expected_substring']}")
        print(f"Before sleep: {objects_before}")
        print(f"After sleep:  {objects_after}")
        print(f"Top type after sleep: {top_type_after}")
        print(f"Sleep report: promoted={sleep_report.get('promoted', 0)}, "
              f"merged={sleep_report.get('merged', 0)}, pruned={sleep_report.get('pruned', 0)}")
        print(f"Pass before: {success_before}")
        print(f"Pass after:  {success_after}")

    accuracy_before = passed_before / total if total else 0.0
    accuracy_after = passed_after / total if total else 0.0

    print("\n=== Summary ===")
    print(f"Passed before sleep: {passed_before}/{total}")
    print(f"Accuracy before:     {accuracy_before:.2%}")
    print(f"Passed after sleep:  {passed_after}/{total}")
    print(f"Accuracy after:      {accuracy_after:.2%}")


if __name__ == "__main__":
    run_quality_benchmark()