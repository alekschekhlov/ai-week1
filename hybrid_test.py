# hybrid_test.py
from db import get_conn, setup
from hybrid import hybrid_search, _dense_ids, _lexical_ids

conn = get_conn()
setup(conn)

# Exact-string query: the flag "--to-earliest" lives verbatim in ONE chunk only
q = "--to-earliest"
print("dense only  :", _dense_ids(conn, q, 5))
print("lexical only:", _lexical_ids(conn, q, 5))
print("hybrid (RRF):", hybrid_search(conn, q))