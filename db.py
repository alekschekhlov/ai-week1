# db.py
import numpy as np
import psycopg
import voyageai
from pgvector.psycopg import register_vector
from corpus import embed_corpus, MODEL, DIMS   # Day 2's cached embedder
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()
DSN = "postgresql://postgres:pass@localhost:5432/postgres"

# (text, source) — source is METADATA we can filter on with plain SQL
CORPUS = [
  ("Kafka consumer groups let multiple consumers share a topic's partitions.", "kafka"),
  ("Reset a consumer group offset with kafka-consumer-groups --reset-offsets.", "kafka"),
  ("A Kafka topic is split into partitions for parallelism.", "kafka"),
  ("Redis is an in-memory key-value store often used for caching.", "redis"),
  ("PostgreSQL supports JSONB columns for semi-structured data.", "postgres"),
]


def get_conn():
  conn = psycopg.connect(DSN, autocommit=True)
  conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
  register_vector(conn)          # lets us pass numpy arrays straight in as `vector`
  return conn


def setup(conn):
  conn.execute("DROP TABLE IF EXISTS documents")
  conn.execute(f"""
        CREATE TABLE documents (
            id        BIGSERIAL PRIMARY KEY,
            content   TEXT NOT NULL,
            source    TEXT NOT NULL,          -- metadata, filter with WHERE
            embedding vector({DIMS})          -- {DIMS} floats per row
        )
    """)
  vecs = embed_corpus([c for c, _ in CORPUS])["vectors"]   # cached -> free on repeat
  for content, source in CORPUS:
    conn.execute(
      "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
      (content, source, np.array(vecs[content])),
    )
  # HNSW index. Operator class vector_cosine_ops MUST match the <=> used when querying.
  conn.execute("CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)")


def search(conn, query: str, k: int = 3, source: str | None = None):
  qv = np.array(vo.embed([query], model=MODEL, input_type="query",
                         output_dimension=DIMS).embeddings[0])
  # <=> is cosine DISTANCE; 1 - distance = similarity, to match Day 3's numpy scores.
  sql = "SELECT content, source, 1 - (embedding <=> %s) AS similarity FROM documents"
  params: list = [qv]
  if source:                                  # metadata filter sits right next to vector search
    sql += " WHERE source = %s"
    params.append(source)
  sql += " ORDER BY embedding <=> %s LIMIT %s"   # same operator as the index
  params += [qv, k]
  return conn.execute(sql, params).fetchall()


if __name__ == "__main__":
  conn = get_conn()
  setup(conn)
  for q in ["how do I rewind a consumer group?", "what is a fast in-memory store?"]:
    print(f"\nQ: {q}")
    for content, source, sim in search(conn, q):
      print(f"  {sim:.3f}  [{source}] {content}")

  # Metadata filter: ask about in-memory store, but restrict to kafka docs only
  print("\nQ (source=kafka only): in-memory store?")
  for content, source, sim in search(conn, "in-memory store?", source="kafka"):
    print(f"  {sim:.3f}  [{source}] {content}")