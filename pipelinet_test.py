# pipeline_test.py
from db import get_conn, setup, search
from rerank import rerank

if __name__ == "__main__":
  conn = get_conn()
  setup(conn)   # re-ingest if needed (cached, cheap)

  query = "how do I rewind a consumer group?"

  # Stage 1: bi-encoder wide net (top 30)
  rows = search(conn, query, k=30)
  candidates = [
    {"source": s, "chunk_index": i, "content": c, "bi_score": round(float(sim), 4)}
    for (s, i, c, sim) in rows
  ]

  # Stage 2: cross-encoder narrows to the best 5
  reranked = rerank(query, candidates, top_k=5)

  print(f"Q: {query}\n")
  for rank, item in enumerate(reranked, 1):
    snippet = item["content"].replace("\n", " ")[:60]
    print(f"{rank}. rerank={item['rerank_score']:.3f}  bi={item['bi_score']:.3f}  "
          f"[{item['source']}#{item['chunk_index']}] {snippet}...")