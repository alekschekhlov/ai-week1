# search.py
import numpy as np
import voyageai
from corpus import embed_corpus, MODEL, DIMS   # reuse Day 2's cached embedder

from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()

CORPUS = [
  "Kafka consumer groups let multiple consumers share a topic's partitions.",
  "Reset a consumer group offset with kafka-consumer-groups --reset-offsets.",
  "A Kafka topic is split into partitions for parallelism.",
  "Redis is an in-memory key-value store often used for caching.",
  "PostgreSQL supports JSONB columns for semi-structured data.",
]


def build_index(texts: list[str]) -> tuple[list[str], np.ndarray]:
  # Embed corpus (cached from Day 2), stack into one matrix: one row per document.
  result = embed_corpus(texts)
  matrix = np.array([result["vectors"][t] for t in texts])   # shape: (n_docs, dims)
  return texts, matrix


def search(query: str, docs: list[str], matrix: np.ndarray, k: int = 3):
  # Query side uses input_type="query" — the retrieval axis from Day 1
  qvec = np.array(
    vo.embed([query], model=MODEL, input_type="query",
             output_dimension=DIMS).embeddings[0]
  )
  # Voyage vectors are unit-length -> dot product == cosine (Day 2).
  # matrix @ qvec computes ALL dot products at once: this single line IS brute-force k-NN.
  scores = matrix @ qvec                      # one score per document
  top = np.argsort(scores)[::-1][:k]          # indices of the k highest scores
  return [(docs[i], float(scores[i])) for i in top]


if __name__ == "__main__":
  docs, matrix = build_index(CORPUS)
  for q in ["how do I rewind a consumer group?", "what is a fast in-memory store?"]:
    print(f"\nQ: {q}")
    for text, score in search(q, docs, matrix):
      print(f"  {score:.3f}  {text}")