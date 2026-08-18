# pick_chunks.py — sample chunks to write golden-set questions against.
#
# Two filters matter. Spread: take chunks evenly across a document instead of the first N,
# so the questions cover the whole page rather than its introduction. Signal: drop chunks
# that are mostly console output or raw data (kubectl tables, the first 2000 digits of pi)
# — they carry no question a user would ask.
import argparse

import psycopg

from db import DSN


def alpha_ratio(text: str) -> float:
  return sum(ch.isalpha() or ch.isspace() for ch in text) / max(len(text), 1)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("docs", nargs="+", help="doc_id values, each optionally as doc:N")
  ap.add_argument("--width", type=int, default=250, help="preview characters per chunk")
  ap.add_argument("--min-alpha", type=float, default=0.75)
  args = ap.parse_args()

  conn = psycopg.connect(DSN, autocommit=True)
  for spec in args.docs:
    doc, _, n = spec.partition(":")
    n = int(n) if n else 10
    rows = conn.execute(
      "SELECT id, section, content FROM chunks WHERE doc_id = %s ORDER BY chunk_index",
      (doc,),
    ).fetchall()
    good = [r for r in rows if alpha_ratio(r[2]) > args.min_alpha]
    step = max(1, len(good) // n)
    sel = good[::step][:n]
    print(f"##### {doc}: {len(rows)} chunks, {len(good)} with prose, taking {len(sel)}")
    for cid, sec, content in sel:
      print(f"-- id={cid} | {sec}")
      print("   " + content[:args.width].replace("\n", " "))


if __name__ == "__main__":
  main()
