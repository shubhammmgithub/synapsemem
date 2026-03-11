import os
from dotenv import load_dotenv
from synapsemem import SynapseManager, FactExtractor

load_dotenv() # Load your OPENAI_API_KEY from .env

# 1. Initialize the system
extractor = FactExtractor(api_key=os.getenv("OPENAI_API_KEY"))
manager = SynapseManager()

# 2. Simulate the "First Message" (The Anchor)
first_msg = "I am building a memory project using Python and ChromaDB. It must avoid hallucinations."
fact_data = extractor.process_text(first_msg)

manager.remember(
    text=fact_data['fact'], 
    topic=fact_data['topic'], 
    priority=10,  # High priority because it's the mission
    is_anchor=True # PINNED!
)

# 3. Test Retrieval
context = manager.get_context("Tell me about my project constraints.")
print(f"LLM Context Found: {context}")

# 4. Test Surgical Deletion (Your unique feature)
manager.storage.delete_topic("Tech")
print("Topic 'Tech' purged. Hallucination risk neutralized.")