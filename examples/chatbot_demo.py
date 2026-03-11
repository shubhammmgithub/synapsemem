"""Example: Chatbot demonstration with SynapseMem"""

"""
Example chatbot using SynapseMem.

Run:
python examples/chatbot_demo.py
"""

from synapsemem import SynapseMemory


def fake_llm(prompt: str) -> str:
    """
    Simple demo LLM function.

    Replace this with OpenAI, Groq, Ollama, etc.
    Example:
        return openai_client(prompt)
    """
    return f"\n[LLM RESPONSE SIMULATION]\n\nPrompt received:\n{prompt}\n"


def main():
    memory = SynapseMemory(
        llm=fake_llm,
        pinned_facts=[
            "You are a helpful AI assistant.",
            "Prefer clear and concise responses."
        ]
    )

    print("SynapseMem Demo Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        response = memory.chat(user_input)

        print("\nAssistant:")
        print(response)
        print("\n---\n")


if __name__ == "__main__":
    main()