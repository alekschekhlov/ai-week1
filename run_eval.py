# run_eval.py — retrieval quality: bi-encoder alone vs bi-encoder + cross-encoder rerank
import numpy as np
import voyageai

from bootstrap import fresh_conn
from corpus import MODEL, DIMS
from eval import evaluate
from generate import _label
from rerank import rerank

vo = voyageai.Client()

# Golden set: query -> the section labels that actually answer it.
# Keyed on LABELS, not on chunk ids: a label survives re-chunking and re-ingesting, while
# integer PKs and chunk_index positions are reassigned on every run.
GOLDEN = {
  "how do I rewind a consumer group?":   {"kafka.md > Kafka > Offsets"},
  "what causes a consumer rebalance?":   {"kafka.md > Kafka > Rebalancing"},
  "what is a fast in-memory cache?":     {"redis.md > Redis > In-Memory Storage"},
  "can Postgres store JSON?":            {"postgres.md > PostgreSQL > JSONB"},
  "how does Postgres do vector search?": {"postgres.md > PostgreSQL > Vector Search with pgvector"},
}


def main():
  conn = fresh_conn()

  def dense(query: str, k: int) -> list[dict]:
    # Bi-encoder wide net. Selects exactly the fields _label() needs, so both retrievers
    # below can score against labels instead of positional ids.
    qv = np.array(vo.embed([query], model=MODEL, input_type="query",
                           output_dimension=DIMS).embeddings[0])
    rows = conn.execute(
      "SELECT doc_id, section, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
      (qv, k),
    ).fetchall()
    return [{"doc_id": d, "section": s, "content": c} for (d, s, c) in rows]

  def bi_only(query):
    return [_label(c) for c in dense(query, 5)]

  def bi_plus_rerank(query):
    # Wide net (30) -> cross-encoder -> top 5. rerank() passes each dict through, so
    # doc_id/section survive and the labels still line up.
    return [_label(c) for c in rerank(query, dense(query, 30), top_k=5)]

  print("bi-only     :", {k: round(v, 3) for k, v in evaluate(bi_only, GOLDEN).items()})
  print("bi + rerank :", {k: round(v, 3) for k, v in evaluate(bi_plus_rerank, GOLDEN).items()})


if __name__ == "__main__":
  main()
