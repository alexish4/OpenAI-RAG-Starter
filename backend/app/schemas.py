from pydantic import BaseModel, Field
from typing import List, Optional

class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    top_k: int = 5

class SourceChunk(BaseModel):
    doc_id: str
    chunk_id: int
    source_name: str
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]