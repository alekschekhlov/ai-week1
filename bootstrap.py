# bootstrap.py — shared fixture for the test/eval scripts.
#
# Replaces the old db.setup(): schema creation lives in db.py, ingestion lives in
# ingest.py, and this glues the two over the sample corpus so every script starts from
# the same known state.
import os

from db import get_conn, setup_schema
from ingest import ingest

# Absolute, so the scripts work regardless of the cwd they are launched from
CORPUS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_corpus")
SOURCE = "docs"


def fresh_conn():
  """Drop, recreate and re-ingest the sample corpus; return an open connection.

  setup_schema() drops the table, so this is destructive by design — the scripts want a
  reproducible corpus. Embeddings are content-addressed in embeddings_cache.json, so only
  the very first run costs anything.
  """
  conn = get_conn()
  setup_schema(conn)
  ingest(conn, CORPUS_ROOT, source=SOURCE)
  return conn
