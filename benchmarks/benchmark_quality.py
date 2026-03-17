from synapsemem import SynapseMemory
from benchmarks.sample_data import QUALITY_TEST_CASES


def run_quality_benchmark():
    passed = 0
    total = len(QUALITY_TEST_CASES)

    print("\n=== Retrieval Quality Benchmark ===")

    for i, case in enumerate(QUALITY_TEST_CASES, start=1):
        memory = SynapseMemory()
        memory.ingest(case["memory"])

        results = memory.retrieve(case["query"], top_k=3)
        objects = " | ".join(result["object"] for result in results)

        success = case["expected_substring"].lower() in objects.lower()
        if success:
            passed += 1

        print(f"\nTest {i}")
        print(f"Memory:   {case['memory']}")
        print(f"Query:    {case['query']}")
        print(f"Expected: {case['expected_substring']}")
        print(f"Retrieved objects: {objects}")
        print(f"Pass: {success}")

    accuracy = passed / total if total else 0.0
    print("\n=== Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    run_quality_benchmark()