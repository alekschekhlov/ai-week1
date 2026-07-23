from functools import lru_cache

from anthropic import AsyncAnthropic

from config import Settings


@lru_cache
def get_client(api_key: str) -> AsyncAnthropic:
    # One shared AsyncAnthropic per API key. The client wraps an httpx pool with
    # keep-alive connections, so reusing it avoids re-doing DNS+TCP+TLS per call.
    # Safe for concurrent use across asyncio tasks.
    return AsyncAnthropic(api_key=api_key)


async def ask_once(settings: Settings, prompt: str) -> str:
    # async client — plays nicely with FastAPI and asyncio.gather
    client = get_client(settings.anthropic_api_key)

    # the core call: model + token cap + a list of messages
    response = await client.messages.create(
        model=settings.default_model,
        max_tokens=settings.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    # response.content is a list of blocks; for plain text, take the first block's text
    return response.content[0].text

async def ask_streaming(settings: Settings, prompt: str) -> str:
    client = get_client(settings.anthropic_api_key)

    collected = []
    # stream() opens an async context manager over the response
    async with client.messages.stream(
        model=settings.default_model,
        max_tokens=settings.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        # text_stream yields text chunks as they arrive
        async for chunk in stream.text_stream:
            print(chunk, end="", flush=True)  # print live, no newline between chunks
            collected.append(chunk)
        print()  # final newline

    return "".join(collected)

# prices in USD per 1,000,000 tokens: (input, output)
# NOTE: verify against current Anthropic pricing — these change over time
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICES[model]
    # scale token counts to millions, then multiply by per-million price
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


async def ask_with_usage(settings: Settings, prompt: str) -> dict:
    client = get_client(settings.anthropic_api_key)

    response = await client.messages.create(
        model=settings.default_model,
        max_tokens=settings.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    # usage carries the real token counts billed for this request
    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens

    return {
        "answer": response.content[0].text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost_usd(settings.default_model, in_tok, out_tok), 6),
    }

# llm.py (add)
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


async def extract_structured(settings: Settings, text: str, schema: type[T]) -> dict:
    # Same shape as ask_with_usage: reuse the shared client, return usage + cost.
    client = get_client(settings.anthropic_api_key)

    response = await client.messages.parse(
        model=settings.default_model,
        max_tokens=settings.max_tokens,
        temperature=0,  # extraction wants determinism, not creativity
        # The prompt is about the TASK; the schema enforces the format (no "return JSON" begging).
        messages=[{"role": "user", "content": f"Extract structured data from this message:\n\n{text}"}],
        output_format=schema,  # SDK derives the JSON schema from your pydantic model
    )

    # Structured outputs still can stop early: refusal or hitting max_tokens → not schema-shaped.
    if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
        raise RuntimeError(f"extraction not schema-shaped (stop_reason={response.stop_reason})")

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    return {
        "result": response.parsed_output,  # already a validated `schema` instance
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost_usd(settings.default_model, in_tok, out_tok), 6),
    }