"""Conversation memory — sliding window with rolling LLM summarisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.llm import LLMClient

SUMMARISE_PROMPT = (
    "Summarise the following conversation so far in a concise paragraph. "
    "Preserve key facts, decisions, tool results, and user intent. "
    "This summary will be used as context for future messages.\n\n"
    "Previous summary:\n{previous_summary}\n\n"
    "New messages to incorporate:\n{messages}"
)


@dataclass
class ConversationMemory:
    """Maintains a sliding window of recent messages plus a rolling summary.

    When the message count exceeds `window_size`, the oldest messages
    beyond the window are summarised and merged into `summary`.
    """

    window_size: int = 10
    summary: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add(self, message: dict[str, Any]) -> None:
        """Append a message to history."""
        self.messages.append(message)

    def needs_compaction(self) -> bool:
        """Check if the history has grown beyond the window."""
        return len(self.messages) > self.window_size

    def compact(self, llm: LLMClient) -> None:
        """Summarise overflow messages and shrink history to the window.

        Args:
            llm: The LLM client used to generate the summary.
        """
        if not self.needs_compaction():
            return

        # Split: messages to summarise vs messages to keep
        overflow_count = len(self.messages) - self.window_size
        to_summarise = self.messages[:overflow_count]
        self.messages = self.messages[overflow_count:]

        # Format overflow messages for the summariser
        formatted = "\n".join(
            f"[{m.get('role', '?')}]: {_extract_text(m)}"
            for m in to_summarise
        )

        prompt = SUMMARISE_PROMPT.format(
            previous_summary=self.summary or "(none)",
            messages=formatted,
        )

        response = llm.chat(
            system="You are a precise conversation summariser.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )

        self.summary = response.text or self.summary

    def get_context(self) -> tuple[str, list[dict[str, Any]]]:
        """Return the current summary and windowed messages.

        Returns:
            (summary, messages) — ready to inject into the system prompt
            and message list respectively.
        """
        return self.summary, list(self.messages)

    def clear(self) -> None:
        """Reset all memory."""
        self.summary = ""
        self.messages.clear()


def _extract_text(message: dict[str, Any]) -> str:
    """Pull plain text from a message dict, handling various content shapes."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    # Anthropic-style content blocks
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool_result: {block.get('content', '')}]")
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_call: {block.get('name', '?')}]")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)
