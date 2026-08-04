# chunk.py
import tiktoken

# Approximate token counter (OpenAI's cl100k; close enough to SIZE Voyage chunks)
_enc = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
  return len(_enc.encode(text))


# Try the LARGEST natural boundary first, fall back to finer ones.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split(text: str, max_tokens: int, seps: list[str]) -> list[str]:
  if n_tokens(text) <= max_tokens or not seps:
    return [text]
  sep, finer = seps[0], seps[1:]
  parts = text.split(sep) if sep else list(text)

  chunks, buf = [], ""
  for part in parts:
    candidate = f"{buf}{sep}{part}" if buf else part
    if n_tokens(candidate) <= max_tokens:
      buf = candidate                       # keep packing this chunk
    else:
      if buf:
        chunks.append(buf)                # flush the full chunk
      if n_tokens(part) > max_tokens:
        chunks.extend(_split(part, max_tokens, finer))  # part too big -> go finer
        buf = ""
      else:
        buf = part
  if buf:
    chunks.append(buf)
  return chunks


def _tail(text: str, k: int) -> str:
  # Last k tokens of a chunk, decoded back to text — used for overlap
  toks = _enc.encode(text)
  return _enc.decode(toks[-k:])


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 80) -> list[str]:
  base = _split(text, max_tokens, SEPARATORS)
  if overlap <= 0 or len(base) < 2:
    return base
  # Prepend the tail of the previous chunk to each next chunk (sliding window)
  out = [base[0]]
  for prev, cur in zip(base, base[1:]):
    out.append(f"{_tail(prev, overlap)} {cur}")
  return out


if __name__ == "__main__":
  doc = """Kafka consumer groups let multiple consumers cooperate to read a topic.
Each partition is assigned to exactly one consumer within the group, which is how
Kafka parallelizes consumption across many machines.

When a consumer joins or leaves, Kafka triggers a rebalance: partitions are
reassigned across the surviving consumers. Frequent rebalances hurt throughput,
so tuning session.timeout.ms and max.poll.interval.ms matters in production.

Offsets track how far a group has read. They are committed back to Kafka, either
automatically or manually. To reprocess data, you reset offsets with the
kafka-consumer-groups tool, choosing --to-earliest, --to-latest, or a timestamp."""

  chunks = chunk_text(doc, max_tokens=60, overlap=15)  # tiny numbers to SEE splitting
  for i, c in enumerate(chunks):
    print(f"\n── chunk {i}  ({n_tokens(c)} tokens) ──\n{c}")