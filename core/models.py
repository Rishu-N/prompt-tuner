from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


@dataclass
class ModelConfig:
    model_id: str
    api_key: str
    base_url: Optional[str] = None
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.model_id

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**d)


@dataclass
class ChatResponse:
    text: str
    usage: Optional[dict] = None  # {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}


class ModelClient:
    """Wraps the OpenAI SDK for universal model access."""

    def __init__(self, config: ModelConfig):
        self.config = config
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs)

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send messages and return the assistant reply text."""
        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def chat_with_usage(self, messages: list[dict], temperature: float = 0.7) -> ChatResponse:
        """Send messages and return both the text and token usage."""
        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip()
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return ChatResponse(text=text, usage=usage)


# -----------------------------------------------------------------------
# Reliability wrappers (decorator pattern)
# -----------------------------------------------------------------------

class CachedModelClient(ModelClient):
    """Wraps a ModelClient with an LRU response cache."""

    def __init__(self, inner: ModelClient, cache):
        self.config = inner.config
        self._client = inner._client
        self._inner = inner
        self._cache = cache

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        cached = self._cache.get(self.config.model_id, messages, temperature)
        if cached is not None:
            return cached
        result = self._inner.chat(messages, temperature)
        self._cache.put(self.config.model_id, messages, temperature, result)
        return result

    def chat_with_usage(self, messages: list[dict], temperature: float = 0.7) -> ChatResponse:
        cached = self._cache.get(self.config.model_id, messages, temperature)
        if cached is not None:
            return ChatResponse(text=cached, usage=None)
        resp = self._inner.chat_with_usage(messages, temperature)
        self._cache.put(self.config.model_id, messages, temperature, resp.text)
        return resp


class RetryModelClient(ModelClient):
    """Wraps a ModelClient with retry + exponential backoff."""

    def __init__(self, inner: ModelClient, policy):
        self.config = inner.config
        self._client = inner._client
        self._inner = inner
        self._policy = policy

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        from reliability.retry import retry_with_backoff
        @retry_with_backoff(self._policy)
        def _call():
            return self._inner.chat(messages, temperature)
        return _call()

    def chat_with_usage(self, messages: list[dict], temperature: float = 0.7) -> ChatResponse:
        from reliability.retry import retry_with_backoff
        @retry_with_backoff(self._policy)
        def _call():
            return self._inner.chat_with_usage(messages, temperature)
        return _call()


class TrackedModelClient(ModelClient):
    """Wraps a ModelClient with cost/token tracking."""

    def __init__(self, inner: ModelClient, tracker):
        self.config = inner.config
        self._client = inner._client
        self._inner = inner
        self._tracker = tracker

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        resp = self._inner.chat_with_usage(messages, temperature)
        self._tracker.record(self.config.model_id, resp.usage)
        return resp.text

    def chat_with_usage(self, messages: list[dict], temperature: float = 0.7) -> ChatResponse:
        resp = self._inner.chat_with_usage(messages, temperature)
        self._tracker.record(self.config.model_id, resp.usage)
        return resp


def build_client(config: ModelConfig, flags=None, cost_tracker=None) -> ModelClient:
    """
    Build a ModelClient with reliability wrappers based on feature flags.
    When flags is None, returns a plain ModelClient.
    """
    client = ModelClient(config)
    if flags is None:
        return client

    if getattr(flags, "cache_enabled", False):
        from reliability.cache import ResponseCache
        cache = ResponseCache(max_size=getattr(flags, "cache_max_size", 256))
        client = CachedModelClient(client, cache)

    if getattr(flags, "retry_enabled", False):
        from reliability.retry import RetryPolicy
        policy = RetryPolicy(
            max_attempts=getattr(flags, "retry_max_attempts", 3),
            base_delay=getattr(flags, "retry_base_delay", 1.0),
        )
        client = RetryModelClient(client, policy)

    if getattr(flags, "cost_tracking_enabled", False) and cost_tracker is not None:
        client = TrackedModelClient(client, cost_tracker)

    return client
