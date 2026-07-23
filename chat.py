import asyncio

from config import Settings
from llm import cost_usd, get_client


async def chat_loop():
    settings = Settings()
    client = get_client(settings.anthropic_api_key)

    system_prompt = "You are a concise, friendly assistant."

    # this list IS the memory — we append to it and resend it every turn
    history: list[dict] = []

    print("Chat started. Type 'quit' to exit.\n")

    while True:
        user_input = input("you> ")
        if user_input.strip().lower() == "quit":
            break

        # add the user's message to history
        history.append({"role": "user", "content": user_input})

        response = await client.messages.create(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            system=system_prompt,
            messages=history,          # send the ENTIRE conversation every time
        )

        answer = response.content[0].text

        # add the model's reply to history so it's remembered next turn
        history.append({"role": "assistant", "content": answer})

        # show the answer + how many tokens this turn cost
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        cost = cost_usd(settings.default_model, in_tok, out_tok)
        print(f"\nbot> {answer}")
        print(f"     [in={in_tok} out={out_tok} cost=${cost:.6f} | history={len(history)} msgs]\n")


if __name__ == "__main__":
    asyncio.run(chat_loop())