# test_day5.py
import asyncio
from config import Settings
from db import get_conn, setup
from generate import answer_cited, _fetch_context
from faithfulness import faithfulness

from dotenv import load_dotenv

load_dotenv()

async def main():
  conn = get_conn()
  setup(conn)                       # re-ingest (cached, cheap)
  settings = Settings()

  # 1) In-corpus question -> answer with API-verified citations
  q = "how do I rewind a consumer group?"
  result = await answer_cited(settings, conn, q)
  print(f"Q: {q}")
  print(f"A: {result['answer']}\n")
  print("Citations (guaranteed real source text):")
  for c in result["citations"]:
    print(f"  [{c['source']}] cited_text={c['cited_text'][:80]!r}")

  # 2) Faithfulness of that answer, judged against the SAME context
  chunks = _fetch_context(conn, q)
  context = "\n".join(f"[{c['id']}] {c['content']}" for c in chunks)
  f = await faithfulness(settings, context, result["answer"])
  print(f"\nfaithfulness score: {f['score']:.2f}")
  for cl in f["claims"]:
    mark = "OK  " if cl["supported"] else "MISS"
    print(f"  [{mark}] {cl['claim']}")

  # 3) Out-of-corpus -> refusal + EMPTY citations (nothing to cite)
  for q2 in ["what is the airspeed velocity of a swallow?", "who created kafka?"]:
    r = await answer_cited(settings, conn, q2)
    print(f"\nQ: {q2}\nA: {r['answer']}")
    print(f"   citations: {len(r['citations'])}  (expect 0)")


asyncio.run(main())