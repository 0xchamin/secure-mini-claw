"""Agentic loop — prompt → LLM → tool call → observe → repeat."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.context import ContextEngine
from src.core.llm import LLMClient, LLMResponse
from src.core.memory import ConversationMemory
from src.core.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default policy: allow everything (OPA replaces this in Step 5)
#ALLOW_ALL: Callable[[str, dict[str, Any]], bool] = lambda tool_name, args: True
from src.core.policy import AllowAllPolicy, PolicyClient, PolicyContext, PolicyDecision



@dataclass
class LoopConfig:
    """Tunable knobs for the agentic loop."""
    max_iterations: int = 10
    temperature: float = 0.0
    max_tokens: int = 4096


class AgentLoop:
    """Core agentic loop connecting context, LLM, tools, memory, and policy."""

    def __init__(
        self,
        llm: LLMClient,
        context: ContextEngine,
        registry: ToolRegistry,
        memory: ConversationMemory | None = None,
        policy_checker: PolicyClient | None = None,

        config: LoopConfig | None = None,
    ):
        self.llm = llm
        self.context = context
        self.registry = registry
        self.memory = memory or ConversationMemory()
        self.policy_checker = policy_checker or AllowAllPolicy()
        self.policy_context = PolicyContext()

        self.config = config or LoopConfig()

    def _build_system_prompt(self) -> str:
        """Assemble system prompt from context files + conversation summary."""
        base = self.context.build_system_prompt()
        summary, _ = self.memory.get_context()
        if summary:
            base += (
                f"\n\n{'=' * 40}\n"
                f"CONVERSATION SUMMARY\n"
                f"{'=' * 40}\n\n"
                f"{summary}"
            )
        return base

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Look up a tool, check policy, execute, and return result as string."""
        tool = self.registry.get(name)
        if tool is None:
            msg = f"Tool '{name}' not found in registry."
            logger.warning(msg)
            return json.dumps({"error": msg})

        # Policy gate
        if not self.policy_checker(name, arguments):
            msg = f"Policy denied execution of tool '{name}'."
            logger.warning(msg)
            return json.dumps({"error": msg, "denied": True})

        try:
            logger.info(f"Executing tool: {name} with args: {arguments}")
            result = tool.handler(**arguments)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            msg = f"Tool '{name}' raised an error: {e}"
            logger.error(msg)
            return json.dumps({"error": msg})

    def _handle_tool_calls(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Execute all tool calls and return tool-result messages.

        Returns Anthropic-style tool_result content blocks.
        """
        results: list[dict[str, Any]] = []
        for tc in response.tool_calls:
            output = self._execute_tool(tc.name, tc.arguments)
            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": output,
            })
        return results

    def run(self, user_input: str) -> str:
        """Run the agentic loop for a single user turn.

        Args:
            user_input: The user's message.

        Returns:
            The agent's final text response.
        """
        # Add user message to memory
        user_message = {"role": "user", "content": user_input}
        self.memory.add(user_message)

        # Compact memory if needed
        if self.memory.needs_compaction():
            self.memory.compact(self.llm)

        system_prompt = self._build_system_prompt()
        _, messages = self.memory.get_context()
        tools = self.registry.get_llm_schemas()

        for iteration in range(self.config.max_iterations):
            logger.info(f"Loop iteration {iteration + 1}/{self.config.max_iterations}")

            response = self.llm.chat(
                system=system_prompt,
                messages=messages,
                tools=tools if tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            if not response.has_tool_calls:
                # Final text response — store and return
                assistant_message = {"role": "assistant", "content": response.text or ""}
                self.memory.add(assistant_message)
                return response.text or ""

            # Assistant message with tool calls (Anthropic format)
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            assistant_message = {"role": "assistant", "content": assistant_content}
            messages.append(assistant_message)
            self.memory.add(assistant_message)

            # Execute tools and append results
            tool_results = self._handle_tool_calls(response)
            tool_message = {"role": "user", "content": tool_results}
            messages.append(tool_message)
            self.memory.add(tool_message)

        # Exhausted iterations
        fallback = "I've reached my maximum reasoning steps. Here's what I know so far."
        if response.text:
            fallback = f"{fallback}\n\n{response.text}"
        self.memory.add({"role": "assistant", "content": fallback})
        return fallback
