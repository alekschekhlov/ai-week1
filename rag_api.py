# rag_api.py — Week 3 capstone: semantic search service over the pgvector corpus
#              Week 4 Day 6: /ask — the full RAG pipeline behind one endpoint
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from config import Settings
from db import DSN, search as db_search   # reuse Day 4-5 retrieval logic verbatim
from faithfulness import faithfulness
from generate import generate_answer, _fetch_context


def _configure(conn):
  # Every pooled connection must learn the `vector` type before running queries
  register_vector(conn)


@asynccontextmanager
async def lifespan(app: FastAPI):
  # ONE shared pool for the whole app. A DB pool genuinely needs shared lifecycle,
  # so lifespan is the right tool here — unlike a per-call API client.
  pool = ConnectionPool(DSN, min_size=1, max_size=8, configure=_configure, open=False)
  pool.open()
  app.state.pool = pool
  yield
  pool.close()


app = FastAPI(lifespan=lifespan)


def get_conn(request: Request):
  # Check a connection out of the pool for this request; return it automatically after.
  with request.app.state.pool.connection() as conn:
    yield conn


def get_settings() -> Settings:
  return Settings()


class SearchRequest(BaseModel):
  query: str = Field(min_length=1, max_length=1000)
  k: int = Field(default=5, ge=1, le=20)
  source: str | None = None                              # metadata filter (Day 4)
  min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)   # pragmatic floor — see debt


class Hit(BaseModel):
  source: str
  chunk_index: int
  content: str
  similarity: float


class SearchResponse(BaseModel):
  query: str
  hits: list[Hit]
  count: int


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, conn=Depends(get_conn)) -> SearchResponse:   # sync def -> threadpool
  try:
    rows = db_search(conn, req.query, k=req.k, source=req.source)
  except Exception as e:
    raise HTTPException(status_code=502, detail="retrieval failed") from e

  # Return the score so the CALLER can judge relevance — because absolute cutoffs are unreliable.
  hits = [
    Hit(source=s, chunk_index=i, content=c, similarity=round(float(sim), 4))
    for (s, i, c, sim) in rows
    if sim >= req.min_similarity
  ]
  return SearchResponse(query=req.query, hits=hits, count=len(hits))


# ---------------------------------------------------------------------------
# Week 4 Day 6 capstone: /ask — retrieve -> generate -> (optionally) judge
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
  query: str = Field(min_length=1, max_length=1000)
  check_faithfulness: bool = False              # off by default: extra call = cost + latency


class Citation(BaseModel):
  doc_id: str
  section: str | None = None
  label: str
  cited_text: str


class AskResponse(BaseModel):
  answer: str
  citations: list[Citation]
  sources: list[str]
  input_tokens: int
  output_tokens: int
  cost_usd: float
  faithfulness: float | None = None


@app.post("/ask", response_model=AskResponse)
async def ask(
  req: AskRequest,
  request: Request,
  settings: Settings = Depends(get_settings),
) -> AskResponse:
  # async def, because generation and the judge are awaited Anthropic calls.
  pool = request.app.state.pool

  def _retrieve():
    # The retrieval half is SYNC (psycopg + Voyage) — it must not run on the loop.
    with pool.connection() as conn:
      return _fetch_context(conn, req.query)

  try:
    chunks = await asyncio.to_thread(_retrieve)                  # don't block the loop
    result = await generate_answer(settings, req.query, chunks)  # async generation
  except Exception as e:
    raise HTTPException(status_code=502, detail="rag pipeline failed") from e

  faith = None
  if req.check_faithfulness:            # opt-in second model call, judged on the SAME chunks
    context = "\n".join(f"[{c['label']}] {c['content']}" for c in chunks)
    faith = round((await faithfulness(settings, context, result["answer"]))["score"], 3)

  return AskResponse(
    answer=result["answer"],
    citations=[Citation(**c) for c in result["citations"]],
    sources=[c["label"] for c in chunks],   # readable labels, attached in _fetch_context
    input_tokens=result["input_tokens"],
    output_tokens=result["output_tokens"],
    cost_usd=result["cost_usd"],
    faithfulness=faith,
  )