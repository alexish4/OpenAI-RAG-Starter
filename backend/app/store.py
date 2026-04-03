import os
import json
import uuid
from typing import Dict, List, Tuple, Any

import numpy as np
import faiss

class VectorStore:
    """
    FAISS index + JSON metadata.
    Stores:
      - vectors in FAISS
      - metadata list in meta.json (parallel to vector ids)
    """

    def __init__(self, vector_dir: str, dim: int):
        self.vector_dir = vector_dir
        self.dim = dim

        os.makedirs(self.vector_dir, exist_ok=True)

        self.index_path = os.path.join(self.vector_dir, "index.faiss")
        self.meta_path = os.path.join(self.vector_dir, "meta.json")

        self.index = self._load_or_create_index()
        self.meta = self._load_meta()

        # keep in sync: meta length == index.ntotal
        if len(self.meta) != self.index.ntotal:
            raise RuntimeError(
                f"Metadata/index mismatch: meta={len(self.meta)} index={self.index.ntotal}"
            )

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        # cosine similarity via normalized vectors + inner product
        return faiss.IndexFlatIP(self.dim)

    def _load_meta(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        """
        vectors: shape (n, dim) float32
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must be (n,{self.dim}) float32")

        self.index.add(vectors)
        self.meta.extend(metas)
        self._persist()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        doc_id: str = None,
        fetch_k: int = 50,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Returns list of (score, meta).
        If doc_id is provided, only returns chunks from that document.
        """
        if self.index.ntotal == 0:
            return []

        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        k = min(fetch_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue

            meta = self.meta[idx]

            if doc_id is not None and meta.get("doc_id") != doc_id:
                continue

            results.append((float(score), meta))

            if len(results) >= top_k:
                break

        return results

def new_doc_id() -> str:
    return uuid.uuid4().hex