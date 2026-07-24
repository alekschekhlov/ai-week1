from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from anthropic import AsyncAnthropic

from config import Settings
from llm import cost_usd, extract_structured, get_client as get_cached_client
from schemas import SupportTicket

app = FastAPI()


# --- dependency: one validated Settings instance, injected everywhere ---
def get_settings() -> Settings:
    return Settings()


# --- dependency: the shared Anthropic client (cached per API key in llm.py) ---
def get_client(settings: Settings = Depends(get_settings)) -> AsyncAnthropic:
    return get_cached_client(settings.anthropic_api_key)


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
            temperature=req.temperature,  # actually forward it to the API
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
        cost_usd=round(cost_usd(settings.default_model, in_tok, out_tok), 6),
    )

class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class ExtractResponse(BaseModel):
    ticket: SupportTicket           # nested validated model
    input_tokens: int
    output_tokens: int
    cost_usd: float


@app.post("/extract")
async def extract(
    req: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    try:
        out = await extract_structured(settings, req.text, SupportTicket)
    except Exception as e:
        # same discipline as /ask: don't leak internals
        raise HTTPException(status_code=502, detail="extraction failed") from e

    return ExtractResponse(
        ticket=out["result"],
        input_tokens=out["input_tokens"],
        output_tokens=out["output_tokens"],
        cost_usd=out["cost_usd"],
    )

# app.py (add)
from llm import run_with_tools


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)


class AgentResponse(BaseModel):
    answer: str
    turns_used: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_reads: int
    cache_writes: int


@app.post("/agent")
async def agent(
    req: AgentRequest,
    settings: Settings = Depends(get_settings),
) -> AgentResponse:
    try:
        out = await run_with_tools(settings, req.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail="agent run failed") from e
    return AgentResponse(**out)