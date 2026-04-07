from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root is 3 levels up from src/bio_agents/config.py
_REPO_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # FutureHouse / Edison Scientific (required for Crow / Falcon / Robin)
    futurehouse_api_key: str = ""
    futurehouse_api_url: str = "https://api.platform.edisonscientific.com"

    # Anthropic
    anthropic_api_key: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Google
    google_api_key: str = ""

    # Groq (free tier)
    groq_api_key: str = ""

    # Local (Ollama)
    ollama_base_url: str = "http://localhost:11434"

    # Biomni (Stanford snap-stanford general-purpose bio agent)
    # Shared project-level data directory — ~11 GB downloaded on first agent run
    biomni_data_path: str = "./data"

    # AWS Bedrock (optional)
    aws_profile: str = ""
    aws_region: str = "us-west-2"

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        """Resolve relative paths to repo root so they work regardless of cwd."""
        p = Path(self.biomni_data_path)
        if not p.is_absolute():
            self.biomni_data_path = str(_REPO_ROOT / p)
        return self


settings = Settings()
