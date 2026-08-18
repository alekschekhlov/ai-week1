# sweep_rerank.py — how deep should the candidate pool handed to the reranker be?
#
# The fusion is computed ONCE per question at the widest depth, then truncated. So the only
# variable across configurations is how many candidates the cross-encoder gets to look at —
# not the candidates themselves or their order.
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import voyageai

from rerank import rerank
from run_eval import GOLDEN, aggregate, conn, gold_ids, score
from hybrid import hybrid_search

POOLS = [10, 15, 20, 30, 40, 60]
WIDEST = max(POOLS)


def rerank_paced(query, cands, top_k):
  # Voyage bills rerank per document, so a deep pool times 153 questions can cross the
  # 2M tokens/minute ceiling. Back off and retry rather than losing the whole sweep.
  for attempt in range(6):
    try:
      return rerank(query, cands, top_k=top_k)
    except voyageai.error.RateLimitError:
      time.sleep(20 * (attempt + 1))
  raise RuntimeError("rate limited past the retry budget")
TOP_K = 10          # what the reranker returns; metrics are measured at k=5
K = 5


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--pools", type=int, nargs="+", default=POOLS)
  args = ap.parse_args()
  pools = args.pools

  for it in GOLDEN:
    it["_gold"] = gold_ids(it["answer_substring"])

  # Phase 1 — fusion once per question, at the widest depth (DB-bound, sequential).
  print(f"fusing {len(GOLDEN)} questions at depth {WIDEST}…")
  fused = []
  for it in GOLDEN:
    ids = [cid for cid, _ in hybrid_search(conn, it["question"], n=WIDEST, top_k=WIDEST)]
    got = dict(conn.execute(
      "SELECT id, content FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall())
    fused.append([(cid, got.get(cid, "")) for cid in ids])

  # Ceiling: is the answer even IN the pool? Reranking can only reorder what it is given,
  # so this bounds every row of the table below.
  print(f"\npool recall (answer present anywhere in the pool, before reranking):")
  for p in pools:
    present = sum(
      any(_hit(c, it) for _cid, c in cand[:p]) for cand, it in zip(fused, GOLDEN))
    print(f"  depth {p:>3}: {present/len(GOLDEN):.3f}")

  # Phase 2 — rerank each (question, depth) pair. Pure HTTP, so run it concurrently.
  print(f"\nreranking {len(GOLDEN)*len(pools)} (question, depth) pairs…")
  print(f"\n{'pool':>5}  " + "  ".join(f"{m:>9}" for m in
                                       (f"recall@{K}", "mrr", f"ndcg@{K}", f"hit@{K}")))
  for p in pools:
    def run(i):
      cand = fused[i][:p]
      out = rerank_paced(GOLDEN[i]["question"],
                   [{"id": cid, "content": c} for cid, c in cand], top_k=TOP_K)
      ranked = [(c["id"], c["content"]) for c in out]
      return score(ranked, GOLDEN[i]["answer_substring"], len(GOLDEN[i]["_gold"]), K)

    with ThreadPoolExecutor(max_workers=(4 if p <= 20 else 2)) as ex:
      rows = list(ex.map(run, range(len(GOLDEN))))
    m = aggregate(rows, K)
    print(f"{p:>5}  " + "  ".join(f"{m[key]:>9.3f}" for key in
                                  (f"recall@{K}", "mrr", f"ndcg@{K}", f"hit@{K}")))


def _hit(content, item):
  from run_eval import _norm
  return _norm(item["answer_substring"]) in _norm(content)


if __name__ == "__main__":
  main()
