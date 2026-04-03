import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import IngestResponse, AskRequest, AskResponse, SourceChunk
from .store import VectorStore
from .ingest import ingest_pdf
from .rag import answer_question

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
VECTOR_DIR = os.getenv("VECTOR_DIR", "app/data")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")

app = FastAPI(title="OpenAI RAG Starter")

client = OpenAI(api_key=OPENAI_API_KEY)

_dim_cache = None
_store = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_store() -> VectorStore:
    global _dim_cache, _store
    if _store is not None:
        return _store

    e = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=["dim_check"])
    dim = len(e.data[0].embedding)

    _dim_cache = dim
    _store = VectorStore(vector_dir=VECTOR_DIR, dim=dim)
    return _store

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    store = get_store()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_pdf(
            client=client,
            store=store,
            embed_model=OPENAI_EMBED_MODEL,
            pdf_path=tmp_path,
            source_name=file.filename,
        )
        return IngestResponse(**result)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    store = get_store()

    out = answer_question(
        client=client,
        store=store,
        model=OPENAI_MODEL,
        embed_model=OPENAI_EMBED_MODEL,
        question=req.question,
        top_k=req.top_k,
        doc_id=req.doc_id,
    )

    sources = [SourceChunk(**s) for s in out["sources"]]
    return AskResponse(answer=out["answer"], sources=sources)