"""Command line interface - 'synapsemem init' and other CLI commands"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from synapsemem import SynapseMemory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synapsemem",
        description="SynapseMem CLI - lightweight neuro-symbolic memory engine",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest text into memory")
    ingest_parser.add_argument("text", type=str, help="Text to ingest")

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve memories for a query")
    retrieve_parser.add_argument("query", type=str, help="Query text")
    retrieve_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    anchor_add_parser = subparsers.add_parser("add-anchor", help="Add a pinned anchor")
    anchor_add_parser.add_argument("text", type=str, help="Anchor text")

    anchor_list_parser = subparsers.add_parser("list-anchors", help="List current anchors")

    reset_parser = subparsers.add_parser("reset", help="Reset all memory state")

    graph_parser = subparsers.add_parser("graph-facts", help="Show graph facts about an entity")
    graph_parser.add_argument("entity", type=str, help="Entity name")

    return parser


def get_memory() -> SynapseMemory:
    """
    MVP CLI uses an in-process memory instance.
    Note: state exists only for the current process run.

    In version B, this should connect to persistent storage.
    """
    return SynapseMemory()


def handle_ingest(memory: SynapseMemory, text: str) -> None:
    memory.ingest(text)
    print("Ingested successfully.")


def handle_retrieve(memory: SynapseMemory, query: str, top_k: int) -> None:
    results = memory.retrieve(query, top_k=top_k)
    print(json.dumps(results, indent=2))


def handle_add_anchor(memory: SynapseMemory, text: str) -> None:
    memory.add_anchor(text)
    print("Anchor added successfully.")


def handle_list_anchors(memory: SynapseMemory) -> None:
    anchors = memory.get_anchors()
    print(json.dumps(anchors, indent=2))


def handle_reset(memory: SynapseMemory) -> None:
    memory.reset()
    print("Memory reset successfully.")


def handle_graph_facts(memory: SynapseMemory, entity: str) -> None:
    facts = memory.graph_facts_about(entity)
    print(json.dumps(facts, indent=2))


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    memory = get_memory()

    if args.command == "ingest":
        handle_ingest(memory, args.text)
    elif args.command == "retrieve":
        handle_retrieve(memory, args.query, args.top_k)
    elif args.command == "add-anchor":
        handle_add_anchor(memory, args.text)
    elif args.command == "list-anchors":
        handle_list_anchors(memory)
    elif args.command == "reset":
        handle_reset(memory)
    elif args.command == "graph-facts":
        handle_graph_facts(memory, args.entity)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()