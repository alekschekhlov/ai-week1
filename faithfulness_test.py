# faithfulness_test.py — API-verified citations + the faithfulness judge
import asyncio

from dotenv import load_dotenv

from bootstrap import fresh_conn
from config import Settings
from faithfulness import faithfulness
from generate import answer_cited, _fetch_context

load_dotenv()


async def main():
  conn = fresh_conn()
  settings = Settings()

  # 1) In-corpus question -> answer with API-verified citations
  q = "how do I rewind a consumer group?"
  result = await answer_cited(settings, conn, q)
  print(f"Q: {q}")
  print(f"A: {result['answer']}\n")
  print("Citations (guaranteed real source text):")
  for c in result["citations"]:
    print(f"  [{c['label']}] cited_text={c['cited_text'][:80]!r}")

  # 2) Faithfulness of that answer, judged against the SAME context
  chunks = _fetch_context(conn, q)
  context = "\n".join(f"[{c['label']}] {c['content']}" for c in chunks)
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


if __name__ == "__main__":
  asyncio.run(main())
