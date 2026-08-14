# generate.py
from anthropic import AsyncAnthropic

from config import Settings
from llm import cost_usd
from hybrid import hybrid_search
from rerank import rerank

RAG_SYSTEM_CITED = """You answer questions using ONLY the provided documents.
If the documents don't contain the answer, say exactly:
"I don't have that information in the provided documents." Do not guess.
Keep answers under 100 words."""


def _label(chunk: dict) -> str:
  # Human-readable citation label: "pods.md > Pod Lifecycle", or just "pods.md" if no section
  return f"{chunk['doc_id']} > {chunk['section']}" if chunk.get("section") else chunk["doc_id"]


def _fetch_context(conn, query: str, top_k: int = 5) -> list[dict]:
  # Stage 1-2: hybrid retrieve (wide net) -> rerank (narrow). Key on the DB primary key `id`.
  fused = hybrid_search(conn, query, n=30, top_k=15)     # [(id, rrf_score)]
  ids = [cid for cid, _score in fused]
  if not ids:
    return []

  rows = conn.execute(
    "SELECT id, doc_id, section, content FROM chunks WHERE id = ANY(%s)", (ids,)
  ).fetchall()
  by_id = {r[0]: {"id": r[0], "doc_id": r[1], "section": r[2], "content": r[3]} for r in rows}

  # preserve the fused ranking order when handing candidates to the reranker
  cands = [by_id[cid] for cid in ids if cid in by_id]
  chunks = rerank(query, cands, top_k=top_k)             # rerank passes each dict through

  # Attach the citation label ONCE, here. Everything downstream — document titles below,
  # /ask's `sources`, the faithfulness context — reads c["label"] instead of recomputing
  # it, so there is exactly one source of truth for how a chunk is named.
  for c in chunks:
    c["label"] = _label(c)
  return chunks


async def generate_answer(settings: Settings, query: str, chunks: list[dict]) -> dict:
  # Generation ONLY (no retrieval) — so the caller can reuse `chunks` for faithfulness.
  documents = [
    {
      "type": "document",
      "source": {"type": "text", "media_type": "text/plain", "data": c["content"]},
      "title": c["label"],                   # readable title -> better citations
      "citations": {"enabled": True},
    }
    for c in chunks
  ]

  client = AsyncAnthropic(api_key=settings.anthropic_api_key)
  resp = await client.messages.create(
    model=settings.default_model,
    max_tokens=settings.max_tokens,
    temperature=0,                             # grounded answers want determinism
    system=RAG_SYSTEM_CITED,
    messages=[{"role": "user", "content": documents + [{"type": "text", "text": query}]}],
  )

  parts, citations = [], []
  for block in resp.content:
    if block.type != "text":
      continue
    parts.append(block.text)
    for cit in (getattr(block, "citations", None) or []):
      src = chunks[cit.document_index]       # document_index maps back to chunks[i]
      citations.append({
        "doc_id": src["doc_id"],
        "section": src["section"],
        "label": src["label"],             # e.g. "pods.md > Pod Lifecycle"
        "cited_text": cit.cited_text,      # guaranteed real source text (API-level)
      })

  in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
  return {
    "answer": "".join(parts),
    "citations": citations,
    "input_tokens": in_tok,
    "output_tokens": out_tok,
    "cost_usd": round(cost_usd(settings.default_model, in_tok, out_tok), 6),
  }


async def answer_cited(settings, conn, query: str) -> dict:
  # convenience wrapper: fetch context, then generate
  return await generate_answer(settings, query, _fetch_context(conn, query))