# Central registry of supported models.
# Each entry: model_key → {provider, model_id, [extra metadata]}
REGISTRY: dict[str, dict] = {
    # Anthropic — litellm_id uses "anthropic/" prefix as required by LiteLLM
    "claude-opus-4-6": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "litellm_id": "anthropic/claude-opus-4-6",
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "litellm_id": "anthropic/claude-sonnet-4-6",
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "litellm_id": "anthropic/claude-haiku-4-5-20251001",
    },
    # OpenAI — no prefix needed by LiteLLM
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o", "litellm_id": "gpt-4o"},
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "litellm_id": "gpt-4o-mini",
    },
    "o4-mini": {"provider": "openai", "model_id": "o4-mini", "litellm_id": "o4-mini"},
    "o4": {"provider": "openai", "model_id": "o4", "litellm_id": "o4"},
    # Google
    "gemini-2.0-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash",
        "litellm_id": "gemini/gemini-2.0-flash",
    },
    "gemini-2.0-pro": {
        "provider": "gemini",
        "model_id": "gemini-2.0-pro",
        "litellm_id": "gemini/gemini-2.0-pro",
    },
    # Groq (free tier — console.groq.com)
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "litellm_id": "groq/llama-3.3-70b-versatile",
    },
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "litellm_id": "groq/llama-3.1-8b-instant",
    },
    # Local (Ollama) — uses ollama_chat/ prefix for chat completions endpoint
    "llama3": {
        "provider": "local",
        "model_id": "llama3",
        "litellm_id": "ollama_chat/llama3",
    },
    "mistral": {
        "provider": "local",
        "model_id": "mistral",
        "litellm_id": "ollama_chat/mistral",
    },
    "qwen3": {
        "provider": "local",
        "model_id": "qwen3",
        "litellm_id": "ollama_chat/qwen3",
    },
}


def get_litellm_id(model_key: str) -> str:
    """Return the LiteLLM-compatible model string for the given registry key."""
    return get_model_info(model_key)["litellm_id"]


def get_model_info(model_key: str) -> dict:
    if model_key not in REGISTRY:
        raise ValueError(f"Unknown model: '{model_key}'. Available: {list(REGISTRY)}")
    return REGISTRY[model_key]


def list_models(provider: str | None = None) -> list[str]:
    if provider:
        return [k for k, v in REGISTRY.items() if v["provider"] == provider]
    return list(REGISTRY)
