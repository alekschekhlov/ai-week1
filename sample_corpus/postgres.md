# PostgreSQL

## JSONB

PostgreSQL is a relational database with strong support for semi-structured
data. The JSONB column type stores JSON in a decomposed binary form, which can be
indexed with GIN indexes for fast containment queries.

## Vector Search with pgvector

With the pgvector extension, PostgreSQL also stores embedding vectors, so you can
run semantic similarity search next to ordinary SQL filters in a single query.
