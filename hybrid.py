# hybrid.py
import numpy as np
import voyageai
from corpus import MODEL, DIMS

vo = voyageai.Client()


def _dense_ids(conn, query, n):
  qv = np.array(vo.embed([query], model=MODEL, input_type="query",
                         output_dimension=DIMS).embeddings[0])
  rows = conn.execute(
    "SELECT source, chunk_index FROM chunks ORDER BY embedding <=> %s LIMIT %s",
    (qv, n),
  ).fetchall()
  return [f"{s}#{i}" for (s, i) in rows]


def _lexical_ids(conn, query, n):
  # ts_rank over the tsvector = Postgres's built-in relevance (BM25-style lexical scoring)
  rows = conn.execute(
    """SELECT source, chunk_index
       FROM chunks
       WHERE fts @@ websearch_to_tsquery('english', %s)
       ORDER BY ts_rank(fts, websearch_to_tsquery('english', %s)) DESC
           LIMIT %s""",
    (query, query, n),
  ).fetchall()
  return [f"{s}#{i}" for (s, i) in rows]


def hybrid_search(conn, query, n=30, k_rrf=60, top_k=5):
  dense = _dense_ids(conn, query, n)
  lexical = _lexical_ids(conn, query, n)

  # Reciprocal Rank Fusion: score by POSITION in each list, scale-independent
  scores: dict[str, float] = {}
  for ranked in (dense, lexical):
    for rank, doc_id in enumerate(ranked, start=1):
      scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)

  fused = sorted(scores, key=scores.get, reverse=True)[:top_k]
  return [(doc_id, round(scores[doc_id], 5)) for doc_id in fused]