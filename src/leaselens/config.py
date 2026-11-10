from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://leaselens:leaselens@localhost:5433/leaselens"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "claude-sonnet-5-5"
    top_k: int = 5
    # Cosine distance above which the best match is treated as unrelated.
    # Calibrated against the eval set in Phase 2.
    max_distance: float = 0.55


@lru_cache
def get_settings() -> Settings:
    return Settings()
