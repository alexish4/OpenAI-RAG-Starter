import re
from typing import List

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking (starter).
    Later: switch to token-based chunking (tiktoken) for better control.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks