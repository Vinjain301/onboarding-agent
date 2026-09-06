"""Chat model access via OpenRouter, with automatic fallback across free
models. Free OpenRouter models share a public capacity pool and are commonly
rate-limited (429) or retired (404) without notice, so a single hardcoded
model is not reliable enough on its own.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompt_values import PromptValue
from openai import APIStatusError

from app import config


def build_chat_model(model: str, temperature: float = 0.1) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
    )


def invoke_with_fallback(prompt_value: PromptValue, temperature: float = 0.1):
    """Invoke the primary chat model, falling back to CHAT_MODEL_FALLBACKS in
    order if the primary is rate-limited or unavailable.

    Returns (response_message, model_name_used) so callers that need a second
    LLM call (e.g. to repair malformed JSON) can reuse the model that worked.
    """
    models = [config.CHAT_MODEL, *config.CHAT_MODEL_FALLBACKS]
    last_error: Exception | None = None

    for model in models:
        try:
            result = build_chat_model(model, temperature).invoke(prompt_value)
            return result, model
        except APIStatusError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"All configured chat models failed (tried: {', '.join(models)}). Last error: {last_error}"
    ) from last_error
