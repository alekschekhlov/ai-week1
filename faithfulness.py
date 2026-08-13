# faithfulness.py — LLM-as-judge (miniature of RAGAS faithfulness)
from anthropic import AsyncAnthropic
from pydantic import BaseModel

JUDGE = """You are a strict fact-checker. Given CONTEXT and an ANSWER, split the answer into
atomic factual claims. For each, decide if the CONTEXT supports it (states or clearly implies it)."""


class Claim(BaseModel):
  claim: str
  supported: bool


class Verdict(BaseModel):
  claims: list[Claim]


async def faithfulness(settings, context: str, answer: str) -> dict:
  client = AsyncAnthropic(api_key=settings.anthropic_api_key)
  resp = await client.messages.parse(
    model=settings.default_model, max_tokens=4096, temperature=0,
    system=JUDGE,
    messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"}],
    output_format=Verdict,   # schema is enforced by the API — no ```json fences, no preamble
  )
  if resp.stop_reason == "max_tokens":                    # truncated -> parsed_output is None
    raise RuntimeError("judge hit max_tokens; raise the limit or shorten the answer")

  claims = resp.parsed_output.claims
  supported = sum(1 for c in claims if c.supported)
  return {
    "score": supported / len(claims) if claims else 1.0,
    "claims": [c.model_dump() for c in claims],           # test prints cl["claim"] / cl["supported"]
  }