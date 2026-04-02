SAMPLE_MEMORIES = [
    "I love hiking.",
    "I work on SynapseMem.",
    "I am preparing for freelancing internship.",
    "My name is Shubham Raj.",
    "I live in Dadri.",
    "I use Python, FastAPI, and ChromaDB.",
    "I prefer automation projects.",
    "I hate cold weather.",
    "I want to build a memory system for LLMs.",
    "I am from Bihar.",
    "I love espresso.",
    "I am interested in AI agents.",
    "I work on long-term memory for chatbots.",
    "I prefer practical project-based learning.",
    "I use VS Code and GitHub daily.",
]

SAMPLE_QUERIES = [
    "What do I love?",
    "What am I working on?",
    "Suggest a project idea for me.",
    "What tools do I use?",
    "Where do I live?",
    "What am I preparing for?",
    "What do I prefer?",
    "What am I interested in?",
]

QUALITY_TEST_CASES = [
    {
        "memory": "I love hiking.",
        "query": "What do I love?",
        "expected_substring": "hiking",
    },
    {
        "memory": "I work on SynapseMem.",
        "query": "What am I working on?",
        "expected_substring": "SynapseMem",
    },
    {
        "memory": "I am preparing for freelancing internship.",
        "query": "What am I preparing for?",
        "expected_substring": "freelancing internship",
    },
    {
        "memory": "I use Python, FastAPI, and ChromaDB.",
        "query": "What tools do I use?",
        "expected_substring": "Python",
    },
    {
        "memory": "I live in Dadri.",
        "query": "Where do I live?",
        "expected_substring": "Dadri",
    },
]

SLEEP_TEST_MEMORIES = [
    "John likes coffee shops.",
    "John likes coffee shops.",
    "John is located in Rome.",
    "John is located in Rome.",
    "I am interested in AI agents.",
    "I am interested in AI agents.",
    "I am in Dadri.",
]

SLEEP_TEST_QUERY = "What does John like?"