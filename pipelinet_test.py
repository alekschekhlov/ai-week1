# pipeline_test.py — two-stage retrieval: bi-encoder wide net -> cross-encoder narrow
from bootstrap import fresh_conn
from db import search
from rerank import rerank


def main():
  conn = fresh_conn()

  query = "how do I rewind a consumer group?"

  # Stage 1: bi-encoder wide net (top 30). db.search() is the Day-4 path still serving
  # /search, so it returns (source, chunk_index, ...) rather than the PK-keyed rows
  # _fetch_context() uses.
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


if __name__ == "__main__":
  main()
