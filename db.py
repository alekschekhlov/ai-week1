# db.py — full RAG retrieval store: chunked ingest + HNSW index + metadata filter
import numpy as np
import psycopg
import voyageai
from pgvector.psycopg import register_vector

from corpus import embed_corpus, MODEL, DIMS   # Day 2's cached embedder
from chunk import chunk_text                    # Day 5's recursive splitter

vo = voyageai.Client()
DSN = "postgresql://postgres:pass@localhost:5432/postgres"

# Longer, realistic docs (not the cheat sentences). (text, source) — source is filterable metadata.
DOCS = [
  ("""Kafka consumer groups let multiple consumers cooperate to read a topic.
Each partition is assigned to exactly one consumer within the group, which is how
Kafka parallelizes consumption across many machines.

When a consumer joins or leaves, Kafka triggers a rebalance: partitions are
reassigned across the surviving consumers. Frequent rebalances hurt throughput,
so tuning session.timeout.ms and max.poll.interval.ms matters in production.

Offsets track how far a group has read. They are committed back to Kafka, either
automatically or manually. To reprocess data, reset offsets with the
kafka-consumer-groups tool, choosing --to-earliest, --to-latest, or a timestamp.""", "kafka"),

  ("""Redis is an in-memory key-value store, so reads and writes are extremely fast
because data lives in RAM rather than on disk. It is most commonly used as a cache
in front of a slower database.

Redis supports rich data types beyond plain strings: lists, sets, sorted sets,
hashes and streams. Persistence is optional via RDB snapshots or the AOF log,
letting you trade durability for speed depending on the use case.""", "redis"),

  ("""PostgreSQL is a relational database with strong support for semi-structured
data. The JSONB column type stores JSON in a decomposed binary form, which can be
indexed with GIN indexes for fast containment queries.

With the pgvector extension, PostgreSQL also stores embedding vectors, so you can
run semantic similarity search next to ordinary SQL filters in a single query.""", "postgres"),
]


def get_conn():
  conn = psycopg.connect(DSN, autocommit=True)
  conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
  register_vector(conn)                        # pass numpy arrays straight in as `vector`
  return conn


def setup(conn):
  conn.execute("DROP TABLE IF EXISTS chunks")
  conn.execute(f"""
        CREATE TABLE chunks (
            id          BIGSERIAL PRIMARY KEY,
            source      TEXT NOT NULL,
            chunk_index INT  NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({DIMS}),
            -- generated tsvector: Postgres keeps the lexical index in sync automatically
            fts         tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
        )
    """)
  for text, source in DOCS:
    _ingest_doc(conn, text, source)
  conn.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)")  # dense
  conn.execute("CREATE INDEX ON chunks USING gin (fts)")                            # lexical (BM25-ish)


def _ingest_doc(conn, text: str, source: str):
  pieces = chunk_text(text, max_tokens=512, overlap=80)   # Day 5 recursive chunking
  vecs = embed_corpus(pieces)["vectors"]                  # cached -> free on repeat runs
  for i, piece in enumerate(pieces):
    conn.execute(
      "INSERT INTO chunks (source, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
      (source, i, piece, np.array(vecs[piece])),
    )


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


if __name__ == "__main__":
  conn = get_conn()
  setup(conn)
  for q in ["how do I rewind a consumer group?",
            "what is a fast in-memory cache?",
            "can Postgres do semantic search?"]:
    print(f"\nQ: {q}")
    for source, idx, content, sim in search(conn, q):
      snippet = content.replace("\n", " ")[:70]
      print(f"  {sim:.3f}  [{source}#{idx}] {snippet}...")

  # Metadata filter demo: ask broadly, restrict to kafka chunks only
  print("\nQ (source=kafka only): how to cache data?")
  for source, idx, content, sim in search(conn, "how to cache data?", source="kafka"):
    snippet = content.replace("\n", " ")[:70]
    print(f"  {sim:.3f}  [{source}#{idx}] {snippet}...")