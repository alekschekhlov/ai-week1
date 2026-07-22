from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # reads variables from .env automatically
    model_config = SettingsConfigDict(env_file=".env")

    anthropic_api_key: str                       # required: app won't start without it
    default_model: str = "claude-haiku-4-5-20251001"  # cheapest model, good for learning
    max_tokens: int = 512