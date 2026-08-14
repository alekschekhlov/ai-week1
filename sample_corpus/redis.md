# Redis

## In-Memory Storage

Redis is an in-memory key-value store, so reads and writes are extremely fast
because data lives in RAM rather than on disk. It is most commonly used as a cache
in front of a slower database.

## Data Types and Persistence

Redis supports rich data types beyond plain strings: lists, sets, sorted sets,
hashes and streams. Persistence is optional via RDB snapshots or the AOF log,
letting you trade durability for speed depending on the use case.
