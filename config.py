from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # reads variables from .env automatically
    # extra="ignore": .env also holds VOYAGE_API_KEY (read by voyageai directly),
    # so don't reject keys that aren't declared here
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str                       # required: app won't start without it
    default_model: str = "claude-haiku-4-5-20251001"  # cheapest model, good for learning
    max_tokens: int = 512