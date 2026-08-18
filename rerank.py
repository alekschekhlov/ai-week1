# rerank.py
import voyageai
from dotenv import load_dotenv

# The client reads VOYAGE_API_KEY at construction, and this module builds one at import
# time — so .env must be loaded HERE. Relying on some other module having called
# load_dotenv() first makes the key depend on import order, which fails silently until
# the first rerank call.
load_dotenv()

vo = voyageai.Client()


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
  """Second pass: a cross-encoder re-scores (query, doc) pairs JOINTLY.

  candidates: dicts with at least a "content" key (the chunk text).
  Returns the top_k candidates, reordered by cross-encoder relevance.
  """
  if not candidates:
    return []

  docs = [c["content"] for c in candidates]
  # Voyage rerank-2.5 scores each (query, doc) pair jointly and returns them sorted
  result = vo.rerank(query, docs, model="rerank-2.5", top_k=top_k)

  out = []
  for r in result.results:
    # r.index points back into the original candidates list; r.relevance_score is the cross-encoder score
    item = dict(candidates[r.index])
    item["rerank_score"] = r.relevance_score
    out.append(item)
  return out