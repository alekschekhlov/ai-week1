# stratify.py — keep golden_set.json proportional to the corpus, and report what's missing.
#
# A golden set built by walking chunk_index ends up concentrated in whichever documents
# happen to sort first, which measures retrieval on a fraction of the corpus. This assigns
# each document a quota proportional to its share of chunks, trims anything over quota,
# and prints the per-document gap so the remaining questions can be aimed at it.
import argparse
import collections
import json
import re

import psycopg

from db import DSN

OUT = "golden_set.json"
STOP = {"what", "which", "when", "where", "does", "with", "from", "that", "this", "have",
        "will", "into", "your", "they", "them", "then", "than", "there", "their", "about",
        "after", "before", "under", "over", "kubernetes", "should", "would", "could", "make"}


def _words(s: str) -> set[str]:
  return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3 and w not in STOP}


def overlap(question: str, content: str) -> float:
  """Share of the question's content words that appear verbatim in the chunk.

  A proxy for how much the question is a restatement of the text it was derived from.
  High overlap makes retrieval look better than it is: real users don't quote the doc.
  """
  qw = _words(question)
  return len(qw & _words(content)) / len(qw) if qw else 0.0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--target", type=int, default=150)
  ap.add_argument("--apply", action="store_true", help="actually trim; otherwise dry-run")
  args = ap.parse_args()

  conn = psycopg.connect(DSN, autocommit=True)
  rows = conn.execute("SELECT id, doc_id, content FROM chunks").fetchall()
  doc_of = {r[0]: r[1] for r in rows}
  content_of = {r[0]: r[2] for r in rows}
  per_doc = collections.Counter(d for _, d, _ in rows)
  total = len(rows)

  doc = json.load(open(OUT, encoding="utf-8"))
  items = doc["items"]

  # Quota: proportional share, but every document gets at least one question so no part
  # of the corpus is invisible to the eval.
  quota = {d: max(1, round(args.target * n / total)) for d, n in per_doc.items()}

  kept, dropped = [], []
  seen: collections.Counter = collections.Counter()
  for it in items:
    d = doc_of.get(it["chunk_id"])
    if d is not None and seen[d] < quota[d]:
      seen[d] += 1
      kept.append(it)
    else:
      dropped.append(it)

  ov = [overlap(it["question"], content_of[it["chunk_id"]]) for it in kept
        if it["chunk_id"] in content_of]
  print(f"target={args.target}  quota sum={sum(quota.values())}  "
        f"have={len(items)} -> keep={len(kept)} drop={len(dropped)}")
  print(f"mean question/chunk word overlap: {sum(ov)/len(ov):.2f}" if ov else "")
  print("\nper-document gap (need = quota - kept):")
  for d, n in per_doc.most_common():
    need = quota[d] - seen[d]
    flag = "  <-- " + ("OVER" if need < 0 else "") if need < 0 else ""
    print(f"  need {need:>3}   have {seen[d]:>3}/{quota[d]:<3} of {n:<4} chunks   {d}{flag}")

  if args.apply:
    doc["items"] = kept
    with open(OUT, "w", encoding="utf-8") as f:
      json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"\ntrimmed {OUT} to {len(kept)} items")
  else:
    print("\n(dry run — pass --apply to trim)")


if __name__ == "__main__":
  main()
