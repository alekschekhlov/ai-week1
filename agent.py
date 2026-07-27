# agent.py
from anthropic import AsyncAnthropic

from config import Settings
from llm import cost_usd
from prompts import AGENT_SYSTEM
from schemas import SupportReply
from tools import TOOLS, execute_tool

# Process-local session store: dies on restart, not shared across workers.
# Production: Redis or Postgres keyed by conversation_id.
_SESSIONS: dict[str, list[dict]] = {}

MAX_TURNS = 5           # circuit breaker for the tool loop
MAX_HISTORY = 20        # naive context guard


def _accumulate(acc: dict, response) -> None:
  """Fold one API response's usage into the running totals for this turn."""
  u = response.usage
  acc["api_calls"] += 1
  acc["input_tokens"] += u.input_tokens
  acc["output_tokens"] += u.output_tokens
  acc["cache_reads"] += getattr(u, "cache_read_input_tokens", 0) or 0
  acc["cache_writes"] += getattr(u, "cache_creation_input_tokens", 0) or 0


async def run_support_turn(
    settings: Settings, conversation_id: str, user_message: str
) -> tuple[SupportReply, dict]:
  client = AsyncAnthropic(api_key=settings.anthropic_api_key)

  # --- Memory: load this conversation and append the incoming turn ---
  history = _SESSIONS.setdefault(conversation_id, [])
  history.append({"role": "user", "content": user_message})

  # Static prefix: tools -> system -> messages. Marker sits at the end of the static part.
  system = [{"type": "text", "text": AGENT_SYSTEM, "cache_control": {"type": "ephemeral"}}]
  acc = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0,
         "cache_reads": 0, "cache_writes": 0}

  # --- PHASE A: gather facts. No schema here, it would ride on every iteration. ---
  for _ in range(MAX_TURNS):
    response = await client.messages.create(
      model=settings.default_model,
      max_tokens=settings.max_tokens,
      temperature=0,                    # support answers want determinism
      system=system,
      tools=TOOLS,
      messages=history,
    )
    _accumulate(acc, response)
    history.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
      break

    results = []
    for block in response.content:
      if block.type != "tool_use":
        continue
      try:
        results.append({
          "type": "tool_result",
          "tool_use_id": block.id,
          "content": execute_tool(block.name, block.input),
        })
      except Exception as e:
        results.append({
          "type": "tool_result",
          "tool_use_id": block.id,
          "content": f"Error: {e}",
          "is_error": True,
        })
    history.append({"role": "user", "content": results})
  else:
    # Loop exhausted with no natural end_turn — fail loudly instead of faking an answer
    raise RuntimeError(f"agent did not converge within {MAX_TURNS} turns")

  # --- PHASE B: structure the final answer. One call, schema paid for once, no tools. ---
  # The instruction below is ephemeral: built as a new list so it never pollutes memory.
  structuring = history + [{
    "role": "user",
    "content": "Produce the final structured support reply for this conversation.",
  }]
  final = await client.messages.parse(
    model=settings.default_model,
    max_tokens=settings.max_tokens,
    temperature=0,
    system=system,
    messages=structuring,
    output_format=SupportReply,
  )
  _accumulate(acc, final)

  if final.stop_reason in ("refusal", "max_tokens") or final.parsed_output is None:
    raise RuntimeError(f"structuring failed (stop_reason={final.stop_reason})")

  # Bound the context: drop the oldest messages once the history gets long
  if len(history) > MAX_HISTORY:
    del history[: len(history) - MAX_HISTORY]

  acc["cost_usd"] = round(
    cost_usd(settings.default_model, acc["input_tokens"], acc["output_tokens"]), 6
  )
  return final.parsed_output, acc