# db.py — full RAG retrieval store: chunked ingest + HNSW index + metadata filter
import numpy as np
import psycopg
import voyageai
from pgvector.psycopg import register_vector

from corpus import MODEL, DIMS                 # Day 2's cached embedder

vo = voyageai.Client()
DSN = "postgresql://postgres:pass@localhost:5432/postgres"


def get_conn():
  conn = psycopg.connect(DSN, autocommit=True)
  conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
  register_vector(conn)                        # pass numpy arrays straight in as `vector`
  return conn


# db.py — setup_schema
def setup_schema(conn):
  conn.execute("DROP TABLE IF EXISTS chunks")
  conn.execute(f"""
        CREATE TABLE chunks (
            id          BIGSERIAL PRIMARY KEY,
            source      TEXT NOT NULL,
            doc_id      TEXT NOT NULL,
            section     TEXT,
            chunk_index INT  NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({DIMS}),
            fts         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            -- Safety net only: retrieval keys on the integer PK `id` (hybrid.py, generate.py),
            -- NOT on this pair. ingest.py numbers chunk_index globally per run, so the pair is
            -- unique by construction — this just makes an accidental double-insert fail loudly.
            UNIQUE (source, chunk_index)
        )
    """)
  # Indexes are created by ingest._ensure_indexes(), after the bulk load.


# db.py — let search fetch a wide net for reranking
def search(conn, query: str, k: int = 30, source: str | None = None):
  # k defaults higher now: bi-encoder casts a WIDE net, reranker narrows it later
  qv = np.array(vo.embed([query], model=MODEL, input_type="query",
                         output_dimension=DIMS).embeddings[0])
  # <=> is cosine DISTANCE (smaller = closer); 1 - distance = similarity, matching Day 3.
  sql = """SELECT source, chunk_index, content, 1 - (embedding <=> %s) AS similarity
           FROM chunks"""
  params: list = [qv]
  if source:                                   # metadata filter next to vector search
    sql += " WHERE source = %s"
    params.append(source)
  sql += " ORDER BY embedding <=> %s LIMIT %s"  # same operator as the index
  params += [qv, k]
  return conn.execute(sql, params).fetchall()