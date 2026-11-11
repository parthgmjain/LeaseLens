from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://leaselens:leaselens@localhost:5433/leaselens"
    # Local sentence-transformers model: free, no API key. bge-small beat MiniLM on
    # deposit-question retrieval in a quick check (see EXPERIMENTS.md once it exists).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # Groq's OpenAI-compatible API; free tier, rate limited.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    # Five chunks: 3 cut §1950.5 from retrieval for the deposit question (see EXPERIMENTS.md).
    top_k: int = 5
    # Hybrid = vector search fused with Postgres full-text search. Set false for vector-only.
    hybrid_search: bool = True
    # Public API limits. They protect the shared free-tier LLM quota from one visitor or a bot.
    ask_limit_per_ip: int = 10
    ask_limit_window_seconds: int = 600
    daily_question_cap: int = 300
    # Cosine distance above which the best match is treated as unrelated.
    # Calibrated against the eval set in Phase 2.
    max_distance: float = 0.55


@lru_cache
def get_settings() -> Settings:
    return Settings()
