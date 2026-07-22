import asyncio

from config import Settings
from llm import ask_with_usage


async def main():
    settings = Settings()  # loads and validates .env at startup
    answer = await ask_with_usage(settings, "Explain what BG deployment is in one sentence.")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())