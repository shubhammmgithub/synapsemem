"""Prompt templates for various tasks"""

SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant.

Use the provided memory carefully:
- Treat pinned anchors as high-priority guidance.
- Use retrieved memories only when they are relevant to the current user request.
- Do not invent facts that are not present in the memory.
- If the memory is insufficient, answer honestly and ask for clarification when needed.
"""

ANCHOR_SECTION_TEMPLATE = """[Pinned Anchors]
{anchors}
"""

MEMORY_SECTION_TEMPLATE = """[Relevant Memory]
{memories}
"""

USER_SECTION_TEMPLATE = """[User Query]
{query}
"""

FULL_PROMPT_TEMPLATE = """{system_prompt}

{anchor_section}

{memory_section}

{user_section}

[Instruction]
Answer the user helpfully, using the memory only when relevant.
"""