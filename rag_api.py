# rag_api.py — Week 3 capstone: semantic search service over the pgvector corpus
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from db import DSN, search as db_search   # reuse Day 4-5 retrieval logic verbatim


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