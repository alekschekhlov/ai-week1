# hybrid.py
import numpy as np
import voyageai
from corpus import MODEL, DIMS

vo = voyageai.Client()


# hybrid.py
def _dense_ids(conn, query, n):
  qv = np.array(vo.embed([query], model=MODEL, input_type="query",
                         output_dimension=DIMS).embeddings[0])
  rows = conn.execute(
    "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (qv, n)
  ).fetchall()
  return [r[0] for r in rows]                      # integer PKs, guaranteed unique

def _lexical_ids(conn, query, n):
  rows = conn.execute(
    """SELECT id FROM chunks
       WHERE fts @@ websearch_to_tsquery('english', %s)
       ORDER BY ts_rank(fts, websearch_to_tsquery('english', %s)) DESC
           LIMIT %s""",
    (query, query, n),
  ).fetchall()
  return [r[0] for r in rows]

def hybrid_search(conn, query, n=30, k_rrf=60, top_k=5):
  dense, lexical = _dense_ids(conn, query, n), _lexical_ids(conn, query, n)
  scores: dict[int, float] = {}
  for ranked in (dense, lexical):
    for rank, cid in enumerate(ranked, start=1):
      scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
  fused = sorted(scores, key=scores.get, reverse=True)[:top_k]
  return [(cid, round(scores[cid], 5)) for cid in fused]   # [(id, rrf_score)]