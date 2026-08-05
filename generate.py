# generate.py
from anthropic import AsyncAnthropic

from config import Settings
from db import get_conn                       # or accept a pooled conn
from hybrid import hybrid_search
from rerank import rerank

RAG_SYSTEM = """You answer questions using ONLY the provided context.

<rules>
- Use only facts from the <context> block. Never use outside knowledge.
- After each claim, cite the source in brackets, e.g. [kafka#0].
- If the context does not contain the answer, say exactly:
  "I don't have that information in the provided documents." Do not guess.
- Keep answers under 100 words, plain sentences.
</rules>"""


def _fetch_context(conn, query: str, top_k: int = 5) -> list[dict]:
  # Stage 1-2: hybrid retrieval, then cross-encoder rerank to the best top_k
  fused = hybrid_search(conn, query, n=30, top_k=15)      # wide net, ids + rrf score
  # hydrate only the fused ids instead of scanning the table; match on the
  # reconstructed "source#chunk_index" so we can pass the ids as a Postgres array
  fused_ids = [fid for (fid, _score) in fused]
  rows = conn.execute(
    "SELECT source, chunk_index, content FROM chunks "
    "WHERE source || '#' || chunk_index = ANY(%s)",
    (fused_ids,),
  ).fetchall()
  by_id = {f"{s}#{i}": {"id": f"{s}#{i}", "content": c} for (s, i, c) in rows}
  cands = [by_id[fid] for (fid, _score) in fused if fid in by_id]   # keep rrf order
  return rerank(query, cands, top_k=top_k)


def _build_prompt(query: str, chunks: list[dict]) -> str:
  context = "\n".join(f"[{c['id']}] {c['content']}" for c in chunks)
  return f"<context>\n{context}\n</context>\n\nQuestion: {query}"


async def answer(settings: Settings, conn, query: str) -> dict:
  chunks = _fetch_context(conn, query)
  user_prompt = _build_prompt(query, chunks)

  client = AsyncAnthropic(api_key=settings.anthropic_api_key)
  resp = await client.messages.create(
    model=settings.default_model,
    max_tokens=settings.max_tokens,
    temperature=0,                          # grounded answers want determinism
    system=RAG_SYSTEM,
    messages=[{"role": "user", "content": user_prompt}],
  )
  text = "".join(b.text for b in resp.content if b.type == "text")
  return {
    "answer": text,
    "sources": [c["id"] for c in chunks],   # what we actually gave the model
    "input_tokens": resp.usage.input_tokens,
    "output_tokens": resp.usage.output_tokens,
  }