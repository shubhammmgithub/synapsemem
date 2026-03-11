"""Text tokenization utilities"""
import re

def simple_tokenize(text: str) -> int:
    """Rough token approximation."""
    tokens = re.findall(r"\w+|\S", text)
    return len(tokens)