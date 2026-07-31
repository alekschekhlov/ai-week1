# corpus.py
import hashlib, json, os
import voyageai
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()
MODEL = "voyage-4-lite"
DIMS = 1024
BATCH = 128                      # safely under Voyage's 1000-string / token caps
CACHE_PATH = "embeddings_cache.json"


def _key(text: str) -> str:
  # Content-addressed: same (model, dims, text) -> same key. Change any field -> new key.
  return hashlib.sha256(f"{MODEL}|{DIMS}|{text}".encode()).hexdigest()


def _load() -> dict:
  return json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def embed_corpus(texts: list[str]) -> dict:
  cache = _load()
  vectors, to_embed, hits = {}, [], 0

  for t in texts:                          # split into cache hits vs misses
    k = _key(t)
    if k in cache:
      vectors[t], hits = cache[k], hits + 1
    else:
      to_embed.append(t)

  billed = 0
  for i in range(0, len(to_embed), BATCH):   # batch ONLY the misses
    batch = to_embed[i:i + BATCH]
    resp = vo.embed(batch, model=MODEL, input_type="document", output_dimension=DIMS)
    billed += resp.total_tokens
    for text, emb in zip(batch, resp.embeddings):
      cache[_key(text)] = emb
      vectors[text] = emb

  json.dump(cache, open(CACHE_PATH, "w"))
  return {
    "vectors": vectors,
    "hits": hits,
    "misses": len(to_embed),
    "billed_tokens": billed,
    "cost_usd": round(billed / 1_000_000 * 0.02, 6),   # voyage-4-lite = $0.02/M
  }


if __name__ == "__main__":
  corpus = [
    "Kafka consumer groups let multiple consumers share a topic's partitions!",
    "Reset a consumer group offset with kafka-consumer-groups --reset-offsets.",
    "A Kafka topic is split into partitions for parallelism.",
    "Redis is an in-memory key-value store often used for caching.",
    "PostgreSQL supports JSONB columns for semi-structured data.",
  ]
  r = embed_corpus(corpus)
  print(f"hits={r['hits']} misses={r['misses']} "
        f"tokens={r['billed_tokens']} cost=${r['cost_usd']}")