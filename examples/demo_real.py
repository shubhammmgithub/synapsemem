"""
SynapseMem — Real Demo
LLM: Groq (Llama3-8b)
Storage: Qdrant Cloud

Setup:
    1. Create .env in project root:
        GROQ_API_KEY=your_groq_key
        QDRANT_URL=https://your-cluster.qdrant.io:6333
        QDRANT_API_KEY=your_qdrant_key

    2. Install deps:
        pip install groq qdrant-client python-dotenv

    3. Run:
        python examples/demo_real.py
"""

import os
import time
from dotenv import load_dotenv

# ── Load credentials from .env ────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL   = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not all([GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise EnvironmentError(
        "Missing credentials. Create a .env file with:\n"
        "  GROQ_API_KEY=...\n"
        "  QDRANT_URL=...\n"
        "  QDRANT_API_KEY=..."
    )

# ── Real LLM: Groq + Llama3 ──────────────────────────────────────────────

try:
    from groq import Groq
except ImportError:
    raise ImportError("Run: pip install groq")

_groq_client = Groq(api_key=GROQ_API_KEY)


def groq_llm(prompt: str) -> str:
    """Calls Groq Llama3-8b. Drop-in replacement for fake_llm."""
    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── SynapseMem setup ──────────────────────────────────────────────────────

from synapsemem import SynapseMemory
from synapsemem.memory.intent_classifier import IntentClassifier
from synapsemem.memory.shared_memory import SharedMemoryStore
from synapsemem.memory.memory_compressor import MemoryCompressor


def section(title: str):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ── Step 1: Initialise ────────────────────────────────────────────────────

def step_init() -> SynapseMemory:
    section("Step 1 — Initialise (Qdrant Cloud + Groq)")

    memory = SynapseMemory(
        llm=groq_llm,
        pinned_facts=[
            "You are a helpful AI assistant with long-term memory.",
            "Use the memory context provided to give personalised responses.",
        ],
        storage_backend="qdrant",
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        user_id="shubham",
        agent_id="assistant",
        session_id="real_demo_001",
    )

    print(f"  LLM      : Groq / llama-3.1-8b-instant")
    print(f"  Storage  : Qdrant Cloud")
    print(f"  URL      : {QDRANT_URL[:45]}...")
    print(f"  Anchors  : {len(memory.get_anchors())} pinned facts")
    return memory


# ── Step 2: Intent-aware ingest ───────────────────────────────────────────

def step_ingest(memory: SynapseMemory):
    section("Step 2 — Intent-Aware Ingest")

    clf = IntentClassifier()

    inputs = [
        "hey",                                                  # chitchat → skip
        "I love hiking and outdoor photography.",               # preference
        "I prefer Python over JavaScript.",                     # preference
        "I am preparing for a machine learning internship.",    # fact
        "I need to finish the SynapseMem demo script today.",   # task
        "I work on SynapseMem — a memory engine for AI agents.",# fact
        "I am from Delhi and interested in agentic AI.",        # fact
        "I prefer dark mode in all my editors.",                # preference
        "I use VSCode and Neovim for development.",             # fact
    ]

    print(f"\n  Ingesting {len(inputs)} inputs into Qdrant Cloud...\n")

    for text in inputs:
        intent = clf.classify(text)

        if clf.should_skip(intent):
            print(f"  [SKIP   ] [{intent:12}] {text}")
            continue

        actions = memory.ingest(text)
        for a in actions:
            t = a["triplet"]
            print(
                f"  [{a['action']:6}] [{intent:12}] "
                f"{t['subject']} {t['predicate']} {t['object']}"
            )

    total = len(memory.storage.all())
    print(f"\n  Active memories in Qdrant: {total}")


# ── Step 3: Real retrieval ────────────────────────────────────────────────

def step_retrieval(memory: SynapseMemory):
    section("Step 3 — Hybrid Retrieval from Qdrant")

    queries = [
        "What are my interests and hobbies?",
        "What tools and languages do I use?",
        "What am I currently working on?",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        results = memory.retrieve(query, top_k=3)
        for r in results:
            print(
                f"    score={r['score']:.3f} | "
                f"{r['subject']} {r['predicate']} {r['object']}"
            )


# ── Step 4: Graph reasoning ───────────────────────────────────────────────

def step_graph(memory: SynapseMemory):
    section("Step 4 — Knowledge Graph")

    print("\n  Facts about 'user':")
    facts = memory.graph_facts_about("user")
    for f in facts[:6]:
        print(f"    user --[{f['predicate']}]--> {f['object']}")

    print(f"\n  Graph nodes total: {len(memory.graph.nodes)}")


# ── Step 5: Real chat with Groq ───────────────────────────────────────────

def step_chat(memory: SynapseMemory):
    section("Step 5 — Real Chat (Groq + Memory Context)")

    questions = [
        "Based on what you know about me, what project should I focus on?",
        "What tools should I use for my internship preparation?",
    ]

    for question in questions:
        print(f"\n  User: {question}")
        print(f"\n  Retrieving memory context...")

        memories = memory.retrieve(question, top_k=5)
        prompt = memory.build_prompt(question, memories)

        print(f"  Calling Groq (llama3-8b-8192)...\n")
        response = groq_llm(prompt)

        print(f"  Assistant:\n")
        # Indent each line for readability
        for line in response.split("\n"):
            print(f"    {line}")

        print()
        time.sleep(0.5)  # be polite to Groq API


# ── Step 6: Memory compression with real LLM ─────────────────────────────

def step_compression(memory: SynapseMemory):
    section("Step 6 — Memory Compression (Groq summarises clusters)")

    # Seed similar memories to create a compressible cluster
    similar = [
        {"subject": "user", "predicate": "likes", "object": "python programming",
         "topic": "preference", "priority": 5, "source_text": "I like python",
         "memory_type": "episodic", "status": "active"},
        {"subject": "user", "predicate": "likes", "object": "coding in python",
         "topic": "preference", "priority": 5, "source_text": "I enjoy coding",
         "memory_type": "episodic", "status": "active"},
        {"subject": "user", "predicate": "likes", "object": "python for ai projects",
         "topic": "preference", "priority": 6, "source_text": "Python for AI",
         "memory_type": "episodic", "status": "active"},
    ]
    memory.storage.add_triplets(similar)

    compressor = MemoryCompressor(
        llm=groq_llm,           # real LLM does the summarisation
        similarity_threshold=0.70,
        min_cluster_size=2,
    )

    print("\n  Dry run — Groq will summarise clusters:")
    report = compressor.run(storage=memory.storage, dry_run=True)
    print(f"    episodic scanned  : {report['episodic_scanned']}")
    print(f"    clusters found    : {report['clusters_found']}")
    print(f"    eligible          : {report['eligible_clusters']}")
    print(f"    would compress    : {report['compressed']}")

    if report["compression_actions"]:
        print(f"\n  Cluster summary (generated by Groq):")
        for action in report["compression_actions"][:2]:
            s = action["summary"]
            print(f"    {s['subject']} {s['predicate']} {s['object']}")
            print(f"    (compressed from {action['cluster_size']} memories)")


# ── Step 7: Multi-agent shared memory ─────────────────────────────────────

def step_shared_memory():
    section("Step 7 — Multi-Agent Shared Memory (SQLite)")

    # Shared memory uses SQLite locally — no Qdrant needed here
    store = SharedMemoryStore(
        workspace_id="demo_team",
        db_path=":memory:",
        conflict_strategy="anchor_weighted",
    )

    agents = {
        "researcher": [
            ("user", "name_is", "Shubham", 9),
            ("user", "located_in", "Delhi", 7),
            ("user", "interested_in", "AI agents", 8),
        ],
        "analyst": [
            ("project", "name_is", "SynapseMem", 9),
            ("project", "phase_is", "Phase 3 complete", 8),
        ],
        "writer": [
            ("user", "prefers", "concise documentation", 5),
        ],
    }

    for agent_id, facts in agents.items():
        for subject, predicate, obj, priority in facts:
            store.write_fact(
                {"subject": subject, "predicate": predicate,
                 "object": obj, "priority": priority},
                agent_id=agent_id,
            )

    stats = store.workspace_stats()
    print(f"\n  Workspace : {stats['workspace_id']}")
    print(f"  Agents    : {stats['contributing_agents']}")
    print(f"  Facts     : {stats['active_records']}")

    print(f"\n  All shared facts:")
    for f in store.read_facts():
        print(
            f"    [{f['agent_id']:12}] "
            f"{f['subject']} {f['predicate']} {f['object']}"
        )


# ── Step 8: Interactive chat loop ─────────────────────────────────────────

def step_interactive(memory: SynapseMemory):
    section("Step 8 — Interactive Chat (type 'exit' to quit)")

    print("\n  Memory is loaded from Qdrant Cloud.")
    print("  Every message you send is ingested and used for retrieval.\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        # Ingest + retrieve + respond
        memory.ingest(user_input)
        memories = memory.retrieve(user_input, top_k=5)
        prompt = memory.build_prompt(user_input, memories)

        print(f"\n  Assistant: ", end="", flush=True)
        response = groq_llm(prompt)
        print(response)
        print()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SynapseMem — Real Demo")
    print("  LLM: Groq (llama3-8b-8192)")
    print("  Storage: Qdrant Cloud")
    print("=" * 60)

    memory = step_init()
    step_ingest(memory)
    step_retrieval(memory)
    step_graph(memory)
    step_chat(memory)
    step_compression(memory)
    step_shared_memory()
    step_interactive(memory)

    print("\n" + "─" * 60)
    print("  Demo complete.")
    print("  Your memories are persisted in Qdrant Cloud.")
    print("  Run again to see retrieval working across sessions.")
    print("─" * 60)


if __name__ == "__main__":
    main()