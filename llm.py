from anthropic import AsyncAnthropic

from config import Settings


async def ask_once(settings: Settings, prompt: str) -> str:
    # async client — plays nicely with FastAPI and asyncio.gather
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # the core call: model + token cap + a list of messages
    response = await client.messages.create(
        model=settings.default_model,
        max_tokens=settings.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    # response.content is a list of blocks; for plain text, take the first block's text
    return response.content[0].text

async def ask_streaming(settings: Settings, prompt: str) -> str:
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

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
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

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