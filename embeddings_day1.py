# embeddings_day1.py
from dotenv import load_dotenv
import voyageai
import numpy as np

load_dotenv()  # loads .env into os.environ

vo = voyageai.Client()  # reads VOYAGE_API_KEY from env

# Three sentences: two mean the same, one is off-topic.
texts = [
  "How do I cancel my subscription?",   # 0
  "I want to unsubscribe from the plan", # 1  — same meaning as 0, different words
  "What time does the store open?",
  "dog",
  "A pet, best friend of human, relative of wolf"# 2  — unrelated
]

# input_type="document" for corpus, "query" for search queries — Voyage optimizes each side
# result = vo.embed(texts, model="voyage-4-lite", input_type="document")
# vecs = [np.array(e) for e in result.embeddings]
#
# print("dims:", len(vecs[0]))          # expect 1024
# print("tokens billed:", result.total_tokens)
#
#
def cosine(a: np.ndarray, b: np.ndarray) -> float:
  # Correct whether or not the vectors are pre-normalized
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

#
# print("0 vs 1 (same meaning): ", round(cosine(vecs[0], vecs[1]), 3))
# print("0 vs 2 (unrelated):    ", round(cosine(vecs[0], vecs[2]), 3))
# print("3 vs 4 (unrelated):    ", round(cosine(vecs[0], vecs[2]), 3))
# print("norm of vec0:          ", round(float(np.linalg.norm(vecs[0])), 3))

q = "A pet, best friend of human, relative of wolf"

# Mode 1 — symmetric, no retrieval prompt (closest to "do these mean the same?")
a = vo.embed(["dog", q], model="voyage-4-lite", input_type=None).embeddings
print("no input_type:", round(cosine(np.array(a[0]), np.array(a[1])), 3))

# Mode 2 — the RETRIEVAL way the model is actually built for
qv = np.array(vo.embed(["What is a dog?"], model="voyage-4-lite", input_type="query").embeddings[0])
docs = ["A pet, best friend of human, relative of wolf",   # relevant
        "The train departs at nine in the morning.",        # irrelevant
        "Interest rates were left unchanged this quarter."]  # irrelevant
dv = [np.array(e) for e in vo.embed(docs, model="voyage-4-lite", input_type="document").embeddings]
scores = [round(cosine(qv, d), 3) for d in dv]
print("retrieval scores:", scores, "→ argmax:", int(np.argmax(scores)))# ~1.0 if normalized