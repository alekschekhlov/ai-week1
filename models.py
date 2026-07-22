from enum import Enum
from pydantic import BaseModel, Field


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class Model(BaseModel):
    name: str = Field(min_length=1)
    provider: Provider
    price_per_1k: float = Field(ge=0)          # cannot be negative
    context_window: int = Field(default=8192, gt=0)

    def cost_for(self, tokens: int) -> float:
        # methods work on BaseModel just like on a dataclass
        return self.price_per_1k * tokens / 1000