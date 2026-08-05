# generate_test.py
import asyncio
from config import Settings
from db import get_conn, setup
from generate import answer


async def main():
  conn = get_conn()
  setup(conn)
  settings = Settings()

  for q in [
    "how do I rewind a consumer group?",     # answerable from kafka#0
    "what is the airspeed velocity of a swallow?",  # NOT in the corpus -> must refuse
    "who created kafka?"
  ]:
    result = await answer(settings, conn, q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print(f"   sources given: {result['sources']}")


asyncio.run(main())