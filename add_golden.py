# add_golden.py — merge hand/agent-authored golden items into golden_set.json.
#
# Companion to make_golden.py for when the API is unavailable: the questions arrive as
# JSONL on stdin instead of from client.messages.create(), but they pass through the SAME
# verbatim check, so both paths produce a set with identical guarantees.
import argparse
import json
import re
import sys

import psycopg

from db import DSN

OUT = "golden_set.json"


def _norm(s: str) -> str:
  return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--update", action="store_true",
                  help="rewrite the question of an item that already exists, instead of "
                       "skipping it as a duplicate (answer_substring is re-checked)")
  args = ap.parse_args()

  conn = psycopg.connect(DSN, autocommit=True)
  rows = conn.execute("SELECT id, doc_id, section, content FROM chunks").fetchall()
  chunks = {r[0]: {"doc_id": r[1], "section": r[2], "content": r[3]} for r in rows}

  try:
    doc = json.load(open(OUT, encoding="utf-8"))
    items = doc["items"]
  except (FileNotFoundError, KeyError, json.JSONDecodeError):
    doc, items = {"source": "k8s", "model": "mixed"}, []
  seen = {it["chunk_id"] for it in items}

  by_id = {it["chunk_id"]: it for it in items}
  added = rejected = dup = updated = 0
  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    rec = json.loads(line)
    cid = rec["chunk_id"]

    if cid in seen and not args.update:
      dup += 1
      continue
    if cid not in chunks:
      print(f"  REJECT {cid}: no such chunk", file=sys.stderr)
      rejected += 1
      continue
    # On --update the substring may be omitted to keep the existing one; a supplied one is
    # still re-checked, so a rephrase can never quietly break the verbatim guarantee.
    if "answer_substring" not in rec and cid in by_id:
      rec["answer_substring"] = by_id[cid]["answer_substring"]

    if _norm(rec["answer_substring"]) not in _norm(chunks[cid]["content"]):
      # Same bar as the API path: a paraphrased "quote" makes the item unverifiable.
      print(f"  REJECT {cid}: substring not verbatim — {rec['answer_substring'][:60]!r}",
            file=sys.stderr)
      rejected += 1
      continue

    if cid in seen:                          # --update: rewrite in place, keep position
      by_id[cid]["question"] = rec["question"]
      by_id[cid]["answer_substring"] = rec["answer_substring"]
      updated += 1
      continue

    item = {
      "question": rec["question"],
      "answer_substring": rec["answer_substring"],
      "chunk_id": cid,
      "doc_id": chunks[cid]["doc_id"],
      "section": chunks[cid]["section"],
    }
    items.append(item)
    by_id[cid] = item
    seen.add(cid)
    added += 1

  doc["items"] = items
  with open(OUT, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
  print(f"added={added} updated={updated} rejected={rejected} duplicate={dup}  "
        f"total={len(items)}/{len(chunks)}")


if __name__ == "__main__":
  main()
