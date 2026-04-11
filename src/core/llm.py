"""BYOK LLM client — unified interface for Anthropic, OpenAI, and Gemini."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class ToolCall:
    """Normalised tool call returned by any provider."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Standardised response from any provider."""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None          # original provider response for debugging
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract LLM client — one per provider."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat request and return a normalised response."""
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """Wraps the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model)
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)

        # Normalise response
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw=resp,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible (covers OpenAI + Gemini)
# ---------------------------------------------------------------------------

class OpenAICompatibleClient(LLMClient):
    """Wraps any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
    ):
        super().__init__(api_key, model)
        import openai
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-style tool defs to OpenAI function-calling format."""
        converted = []
        for tool in tools:
            # If already in OpenAI format, pass through
            if "function" in tool:
                converted.append(tool)
            else:
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                })
        return converted

    def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        import json

        full_messages = [{"role": "system", "content": system}] + messages

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message

        # Normalise tool calls
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            raw=resp,
            usage={
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        )


# ---------------------------------------------------------------------------
# Convenience subclasses
# ---------------------------------------------------------------------------

class OpenAIClient(OpenAICompatibleClient):
    """OpenAI (direct)."""
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key=api_key, model=model)


class GeminiClient(OpenAICompatibleClient):
    """Google Gemini via its OpenAI-compatible endpoint."""
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(
    provider: str | Provider,
    api_key: str,
    model: str | None = None,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: One of 'anthropic', 'openai', 'gemini'.
        api_key: The API key (never read from env — caller controls this).
        model: Optional model override; each provider has a sensible default.

    Returns:
        An LLMClient instance.
    """
    provider = Provider(provider)
    match provider:
        case Provider.ANTHROPIC:
            return AnthropicClient(api_key=api_key, model=model or "claude-sonnet-4-20250514")
        case Provider.OPENAI:
            return OpenAIClient(api_key=api_key, model=model or "gpt-4o")
        case Provider.GEMINI:
            return GeminiClient(api_key=api_key, model=model or "gemini-2.0-flash")
        case _:
            raise ValueError(f"Unknown provider: {provider}")
