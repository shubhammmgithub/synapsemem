from synapsemem import SynapseMemory
from synapsemem.utils.tokenizer import simple_tokenize
from benchmarks.sample_data import SAMPLE_MEMORIES, SAMPLE_QUERIES


def run_prompt_benchmark():
    memory = SynapseMemory(
        pinned_facts=[
            "You are a helpful AI assistant.",
            "Be concise and accurate.",
        ]
    )

    for text in SAMPLE_MEMORIES:
        memory.ingest(text)

    print("\n=== Prompt Benchmark ===")

    for query in SAMPLE_QUERIES:
        results = memory.retrieve(query, top_k=5)
        prompt = memory.build_prompt(query, results)
        token_count = simple_tokenize(prompt)

        print("\n---")
        print(f"Query: {query}")
        print(f"Prompt characters: {len(prompt)}")
        print(f"Approx tokens: {token_count}")


if __name__ == "__main__":
    run_prompt_benchmark()