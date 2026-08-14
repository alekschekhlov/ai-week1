# hybrid_test.py — dense vs lexical vs RRF fusion on an exact-string query
from bootstrap import fresh_conn
from generate import _label                     # one source of truth for chunk naming
from hybrid import hybrid_search, _dense_ids, _lexical_ids


def main():
  conn = fresh_conn()

  def labels(ids):
    # hybrid.py returns integer PKs now; resolve them to readable labels for printing
    if not ids:
      return []
    rows = conn.execute(
      "SELECT id, doc_id, section FROM chunks WHERE id = ANY(%s)", (list(ids),)
    ).fetchall()
    by_id = {r[0]: _label({"doc_id": r[1], "section": r[2]}) for r in rows}
    return [by_id.get(i, f"<missing {i}>") for i in ids]

  # Exact-string query: the flag "--to-earliest" lives verbatim in ONE chunk only
  q = "--to-earliest"
  print(f"Q: {q}\n")
  print("dense only  :", labels(_dense_ids(conn, q, 5)))
  print("lexical only:", labels(_lexical_ids(conn, q, 5)))
  print("hybrid (RRF):", labels([cid for cid, _score in hybrid_search(conn, q)]))


if __name__ == "__main__":
  main()
