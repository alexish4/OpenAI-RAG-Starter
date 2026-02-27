from typing import List, Dict, Any
import os
import numpy as np
from pypdf import PdfReader

from openai import OpenAI
from .utils import chunk_text
from .store import VectorStore, new_doc_id

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def normalize(v: np.ndarray) -> np.ndarray:
    # for cosine similarity with IndexFlatIP
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def embed_texts(client: OpenAI, embed_model: str, texts: List[str]) -> np.ndarray:
    # OpenAI embeddings endpoint via SDK
    resp = client.embeddings.create(
        model=embed_model,
        input=texts,
    )
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return normalize(vecs)

def ingest_pdf(
    client: OpenAI,
    store: VectorStore,
    embed_model: str,
    pdf_path: str,
    source_name: str,
) -> Dict[str, Any]:
    doc_id = new_doc_id()

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    if not chunks:
        return {"doc_id": doc_id, "chunks_added": 0}

    vectors = embed_texts(client, embed_model, chunks)

    metas = []
    for i, chunk in enumerate(chunks):
        metas.append(
            {
                "doc_id": doc_id,
                "chunk_id": i,
                "source_name": source_name,
                "text": chunk,
            }
        )

    store.add(vectors, metas)
    return {"doc_id": doc_id, "chunks_added": len(chunks)}