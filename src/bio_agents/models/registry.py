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
    # Ollama Cloud — remote GPU inference via ollama.com
    # Routed through the local Ollama daemon; the :<size>-cloud tag tells the
    # daemon to offload to Ollama's cloud infrastructure instead of local GPU.
    # Requirements: `ollama serve` + `ollama signin` (one-time CLI login).
    # Full model list: https://ollama.com/search?c=cloud
    # Tags below are verified from individual library pages (May 2026).
    #
    # Subscription tiers (https://ollama.com/pricing):
    #   Low usage    → Free tier  (light quota, 1 concurrent model)
    #   Medium usage → Pro        ($20/month — "larger cloud models")
    #   High usage   → Pro / Max  ($100/month for sustained heavy use)
    #
    # --- Free tier (Low usage) ---
    "ministral-3b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "ministral-3:3b-cloud",
        "litellm_id": "ollama_chat/ministral-3:3b-cloud",
        "cloud_usage": "low",  # free tier
    },
    "ministral-8b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "ministral-3:8b-cloud",
        "litellm_id": "ollama_chat/ministral-3:8b-cloud",
        "cloud_usage": "low",  # free tier
    },
    "ministral-14b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "ministral-3:14b-cloud",
        "litellm_id": "ollama_chat/ministral-3:14b-cloud",
        "cloud_usage": "low",  # free tier
    },
    "gemma34b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "gemma3:4b-cloud",
        "litellm_id": "ollama_chat/gemma3:4b-cloud",
        "cloud_usage": "low",  # free tier
    },
    "gemma312b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "gemma3:12b-cloud",
        "litellm_id": "ollama_chat/gemma3:12b-cloud",
        "cloud_usage": "low",  # free tier
    },
    "gpt-oss20b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "gpt-oss:20b-cloud",
        "litellm_id": "ollama_chat/gpt-oss:20b-cloud",
        "cloud_usage": "low",  # free tier
    },
    "nemotron-3-nano30b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "nemotron-3-nano:30b-cloud",
        "litellm_id": "ollama_chat/nemotron-3-nano:30b-cloud",
        "cloud_usage": "low",  # free tier
    },
    # --- Pro tier required (Medium usage) ---
    "gemma327b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "gemma3:27b-cloud",
        "litellm_id": "ollama_chat/gemma3:27b-cloud",
        "cloud_usage": "medium",  # requires Pro subscription
    },
    "gemma431b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "gemma4:31b-cloud",
        "litellm_id": "ollama_chat/gemma4:31b-cloud",
        "cloud_usage": "medium",  # requires Pro subscription
    },
    "deepseek-v4-flash-cloud": {
        "provider": "ollama_cloud",
        "model_id": "deepseek-v4-flash:cloud",
        "litellm_id": "ollama_chat/deepseek-v4-flash:cloud",
        "cloud_usage": "medium",  # requires Pro subscription
    },
    "qwen3.5-cloud": {
        "provider": "ollama_cloud",
        "model_id": "qwen3.5:cloud",
        "litellm_id": "ollama_chat/qwen3.5:cloud",
        "cloud_usage": "medium",  # requires Pro subscription
    },
    # --- Pro / Max tier required (High usage) ---
    "kimi-k2.6-cloud": {
        "provider": "ollama_cloud",
        "model_id": "kimi-k2.6:cloud",
        "litellm_id": "ollama_chat/kimi-k2.6:cloud",
        "cloud_usage": "high",  # requires Pro or Max subscription
    },
    "glm-4.6-cloud": {
        "provider": "ollama_cloud",
        "model_id": "glm-4.6:cloud",
        "litellm_id": "ollama_chat/glm-4.6:cloud",
        "cloud_usage": "high",  # requires Pro or Max subscription
    },
    "deepseek-v3.1-671b-cloud": {
        "provider": "ollama_cloud",
        "model_id": "deepseek-v3.1:671b-cloud",
        "litellm_id": "ollama_chat/deepseek-v3.1:671b-cloud",
        "cloud_usage": "high",  # requires Pro or Max subscription
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
