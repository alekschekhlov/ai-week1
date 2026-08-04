# run_eval.py
from db import get_conn, setup, search
from rerank import rerank
from eval import evaluate

conn = get_conn()
setup(conn)

# Golden set: query -> which chunk ids actually answer it. Chunk id = f"{source}#{index}".
# NOTE: these ids are tied to your CURRENT chunking config. Re-chunk -> re-label (see debt).
GOLDEN = {
  "how do I rewind a consumer group?": {"kafka#0"},
  "what causes a consumer rebalance?": {"kafka#0"},
  "what is a fast in-memory cache?":   {"redis#0"},
  "can Postgres store JSON?":          {"postgres#0"},
  "how does Postgres do vector search?": {"postgres#0"},
}


def _ids(rows):
  return [f"{s}#{i}" for (s, i, _c, _sim) in rows]


def bi_only(query):
  # Bi-encoder retrieval, top 5
  return _ids(search(conn, query, k=5))


def bi_plus_rerank(query):
  # Wide net (30) -> cross-encoder -> top 5
  rows = search(conn, query, k=30)
  cands = [{"source": s, "chunk_index": i, "content": c} for (s, i, c, _sim) in rows]
  reranked = rerank(query, cands, top_k=5)
  return [f"{c['source']}#{c['chunk_index']}" for c in reranked]


print("bi-only     :", {k: round(v, 3) for k, v in evaluate(bi_only, GOLDEN).items()})
print("bi + rerank :", {k: round(v, 3) for k, v in evaluate(bi_plus_rerank, GOLDEN).items()})