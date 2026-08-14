# generate_test.py — grounded generation: cites when it can, refuses when it can't
import asyncio

from bootstrap import fresh_conn
from config import Settings
from generate import answer_cited


async def main():
  conn = fresh_conn()
  settings = Settings()

  for q in [
    "how do I rewind a consumer group?",             # answerable from kafka.md > Offsets
    "what is the airspeed velocity of a swallow?",   # NOT in the corpus -> must refuse
    "who created kafka?",                            # plausible, but not in the corpus
  ]:
    result = await answer_cited(settings, conn, q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print(f"   citations: {[c['label'] for c in result['citations']] or 'none'}")


if __name__ == "__main__":
  asyncio.run(main())
