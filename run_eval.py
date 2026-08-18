# run_eval.py — compare retrieval strategies against the golden set.
import json
import re

import db
from eval import recall_at_k, reciprocal_rank, ndcg_at_k
from hybrid import hybrid_search
from rerank import rerank

conn = db.get_conn()
GOLDEN = json.load(open("golden_set.json"))["items"]   # the file is {source, model, items:[...]}


def _norm(s: str) -> str:
  # The golden substrings were authored and validated under whitespace normalization —
  # newlines inside a quote were collapsed to spaces. Comparing raw would fail on 61 of
  # 153 items and silently score them as never-relevant, deflating every metric ~40%.
  return re.sub(r"\s+", " ", s).strip().lower()


_CORPUS = [(cid, _norm(c)) for cid, c in
           conn.execute("SELECT id, content FROM chunks").fetchall()]


def gold_ids(answer_substring: str) -> set[int]:
  """Every chunk in the corpus that contains the gold answer.

  This is the denominator recall needs. Usually one chunk; more where the k8s docs repeat
  a sentence across pages, or where a chunk's sliding-window overlap duplicates it.
  """
  s = _norm(answer_substring)
  return {cid for cid, content in _CORPUS if s in content}


def _contents(ids):
  # map retrieved ids -> content, preserving rank order
  fetched = dict(conn.execute(
    "SELECT id, content FROM chunks WHERE id = ANY(%s)", (list(ids),)).fetchall())
  return [(cid, fetched.get(cid, "")) for cid in ids]


def dense_only(query):
  rows = db.search(conn, query, k=10)                  # (source, chunk_index, content, sim)
  return [(None, c) for (_s, _i, c, _sim) in rows]


def hybrid_no_rerank(query):
  fused = hybrid_search(conn, query, n=30, top_k=10)
  return _contents([cid for cid, _ in fused])


def hybrid_rerank(query):
  # Pool depth 30 comes from sweep_rerank.py: recall@5 climbs 0.831 -> 0.894 going from 15
  # to 30, then flattens (0.907 at 40, 0.924 at 60) while rerank cost stays linear in pool
  # size. 30 is the knee.
  fused = hybrid_search(conn, query, n=60, top_k=30)
  cands = [{"id": cid, "content": c} for cid, c in _contents([cid for cid, _ in fused])]
  reranked = rerank(query, cands, top_k=10)
  return [(c["id"], c["content"]) for c in reranked]


def score(ranked, answer_substring, n_gold, k=5):
  """Metrics for one ranked list. `ranked` is [(id, content)], newest-first by rank."""
  s = _norm(answer_substring)
  rel_pos = [i for i, (_cid, content) in enumerate(ranked) if s in _norm(content)]

  # eval.py's metrics take (ranked ids, set of relevant ids). Ranked ids are positions.
  # Relevant chunks the retriever MISSED still belong in the denominator, so pad the
  # relevant set with negative sentinels that can never appear in the ranked list —
  # otherwise recall is |found in top-k| / |found at all|, which is 1.0 by construction.
  missed = max(0, n_gold - len(rel_pos))
  relevant = set(rel_pos) | {-(j + 1) for j in range(missed)}
  ids = list(range(len(ranked)))

  return (recall_at_k(ids, relevant, k), reciprocal_rank(ids, relevant),
          ndcg_at_k(ids, relevant, k), float(any(p < k for p in rel_pos)))


def aggregate(rows, k=5):
  n = len(rows)
  R, M, N, H = (sum(c) / n for c in zip(*rows))
  return {f"recall@{k}": round(R, 3), "mrr": round(M, 3),
          f"ndcg@{k}": round(N, 3), f"hit@{k}": round(H, 3)}


def evaluate(retriever, k=5):
  return aggregate([score(retriever(it["question"]), it["answer_substring"],
                          len(it["_gold"]), k) for it in GOLDEN], k)


if __name__ == "__main__":
  for it in GOLDEN:
    it["_gold"] = gold_ids(it["answer_substring"])
  sizes = [len(it["_gold"]) for it in GOLDEN]
  print(f"{len(GOLDEN)} questions   relevant chunks per question: "
        f"min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.2f}\n")

  for name, fn in [("dense only", dense_only),
                   ("hybrid (no rerank)", hybrid_no_rerank),
                   ("hybrid + rerank", hybrid_rerank)]:
    print(f"{name:22} {evaluate(fn)}")
