from typing import List, Dict, Any
import numpy as np

from openai import OpenAI
from .store import VectorStore
from .ingest import embed_texts

SYSTEM_INSTRUCTIONS = """You are a helpful assistant.
Use ONLY the provided context to answer.
If the answer is not in the context, say: "I don't know based on the provided documents."
Cite sources by chunk_id and source_name when relevant.
"""

def build_context(results: List[tuple]) -> str:
    # results: [(score, meta), ...]
    blocks = []
    for score, meta in results:
        blocks.append(
            f"[source={meta['source_name']} chunk_id={meta['chunk_id']} score={score:.3f}]\n{meta['text']}"
        )
    return "\n\n---\n\n".join(blocks)

def answer_question(
    client: OpenAI,
    store: VectorStore,
    model: str,
    embed_model: str,
    question: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    q_vec = embed_texts(client, embed_model, [question])
    retrieved = store.search(q_vec[0], top_k=top_k)

    context = build_context(retrieved)

    prompt = f"""Context:
{context}

Question:
{question}
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    # SDK returns structured output; simplest: resp.output_text
    answer = getattr(resp, "output_text", None)
    if answer is None:
        # fallback: try to reconstruct if SDK changes
        answer = str(resp)

    sources = [meta for _, meta in retrieved]
    return {"answer": answer, "sources": sources}