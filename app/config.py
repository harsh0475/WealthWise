"""Configuration settings for WealthWise.

Uses pydantic-settings to load settings from environment variables or .env file.
"""
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    llm_provider: Literal["groq", "ollama", "mock"] = "groq"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = "gsk_dummy_key"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    max_agent_iterations: int = 5
    log_level: str = "INFO"

    # Context window token budget safety margin
    max_categorization_batch_tokens: int = 2000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton settings instance
settings = Settings()

