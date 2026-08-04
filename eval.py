# eval.py
import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
  # Of the relevant items that exist, how many are in the top k? (coverage)
  top = retrieved[:k]
  return len(set(top) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
  # 1 / rank of the FIRST relevant hit; 0 if none found
  for i, doc in enumerate(retrieved, start=1):
    if doc in relevant:
      return 1.0 / i
  return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
  def dcg(ids: list[str]) -> float:
    return sum((1.0 if d in relevant else 0.0) / math.log2(i + 1)
               for i, d in enumerate(ids[:k], start=1))
  ideal = dcg(list(relevant))     # relevant items at the top = perfect order
  return dcg(retrieved) / ideal if ideal > 0 else 0.0


def evaluate(retriever, golden: dict[str, set[str]], k: int = 5) -> dict:
  """retriever(query) -> ordered list of chunk ids. golden: query -> set of relevant ids."""
  recalls, rrs, ndcgs = [], [], []
  for query, relevant in golden.items():
    retrieved = retriever(query)          # ranked list of chunk ids
    recalls.append(recall_at_k(retrieved, relevant, k))
    rrs.append(reciprocal_rank(retrieved, relevant))
    ndcgs.append(ndcg_at_k(retrieved, relevant, k))
  n = len(golden)
  return {f"recall@{k}": sum(recalls) / n,
          "mrr": sum(rrs) / n,
          f"ndcg@{k}": sum(ndcgs) / n}