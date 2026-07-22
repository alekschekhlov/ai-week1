from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from anthropic import AsyncAnthropic

from config import Settings
from llm import cost_usd

app = FastAPI()


# --- dependency: one validated Settings instance, injected everywhere ---
def get_settings() -> Settings:
    return Settings()


# --- dependency: a shared Anthropic client built from settings ---
def get_client(settings: Settings = Depends(get_settings)) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


# --- schemas ---
class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)


class AskResponse(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@app.post("/ask")
async def ask(
    req: AskRequest,
    settings: Settings = Depends(get_settings),
    client: AsyncAnthropic = Depends(get_client),
) -> AskResponse:
    try:
        response = await client.messages.create(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            messages=[{"role": "user", "content": req.prompt}],
        )
    except Exception as e:
        # never leak raw internal errors to the client
        raise HTTPException(status_code=502, detail="LLM request failed") from e

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens

    return AskResponse(
        answer=response.content[0].text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        temperature=req.temperature,
        cost_usd=round(cost_usd(settings.default_model, in_tok, out_tok), 6),
    )