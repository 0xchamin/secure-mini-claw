"""CLI — local REPL for interacting with the agent."""

from __future__ import annotations

import argparse
import os
import sys

from src.core.context import ContextEngine
from src.core.llm import create_client
from src.core.loop import AgentLoop, LoopConfig
from src.core.memory import ConversationMemory
from src.core.policy import AllowAllPolicy, OPAClient
from src.core.registry import SQLiteToolRegistry

COMMANDS = {
    "/help":   "Show this help message",
    "/quit":   "Exit the CLI",
    "/clear":  "Clear conversation memory",
    "/tools":  "List registered tools",
    "/memory": "Show current memory summary",
    "/config": "Show active configuration",
}


def print_help() -> None:
    print("\n  Available commands:\n")
    for cmd, desc in COMMANDS.items():
        print(f"    {cmd:<10} {desc}")
    print()


def print_tools(registry: SQLiteToolRegistry) -> None:
    tools = registry.list_tools()
    if not tools:
        print("\n  No tools registered.\n")
        return
    print("\n  Registered tools:\n")
    for t in tools:
        print(f"    [{t.skill}] {t.name} v{t.version} — {t.description}")
    print()


def print_memory(memory: ConversationMemory) -> None:
    summary, messages = memory.get_context()
    print(f"\n  Summary: {summary or '(none)'}")
    print(f"  Window:  {len(messages)}/{memory.window_size} messages\n")


def print_config(provider: str, model: str, opa_url: str | None) -> None:
    print(f"\n  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  Policy:   {'OPA @ ' + opa_url if opa_url else 'AllowAll (dev mode)'}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure Mini Claw — CLI Agent")
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "gemini"],
        default="anthropic", help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: provider's default)",
    )
    parser.add_argument(
        "--api-key-env", default=None,
        help="Env var name holding the API key (default: <PROVIDER>_API_KEY)",
    )
    parser.add_argument(
        "--project-root", default=".",
        help="Path to project root with AGENTS.md etc (default: .)",
    )
    parser.add_argument(
        "--window-size", type=int, default=10,
        help="Conversation memory window size (default: 10)",
    )
    parser.add_argument(
        "--opa-url", default=None,
        help="OPA server URL (default: AllowAll policy)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Max agentic loop iterations per turn (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve API key
    env_var = args.api_key_env or f"{args.provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        print(f"Error: Set {env_var} environment variable with your API key.")
        sys.exit(1)

    # Wire up components
    context = ContextEngine(args.project_root)
    llm = create_client(args.provider, api_key, args.model)
    registry = SQLiteToolRegistry()
    memory = ConversationMemory(window_size=args.window_size)
    policy = OPAClient(opa_url=args.opa_url) if args.opa_url else AllowAllPolicy()
    config = LoopConfig(max_iterations=args.max_iterations)

    agent = AgentLoop(
        llm=llm,
        context=context,
        registry=registry,
        memory=memory,
        policy_checker=policy,
        config=config,
    )

    print("\n  🦀 Secure Mini Claw v0.1.0")
    print("  Type /help for commands.\n")

    while True:
        try:
            user_input = input("  you > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!\n")
            break

        if not user_input:
            continue

        match user_input:
            case "/help":
                print_help()
            case "/quit":
                print("\n  Goodbye!\n")
                break
            case "/clear":
                memory.clear()
                print("\n  Memory cleared.\n")
            case "/tools":
                print_tools(registry)
            case "/memory":
                print_memory(memory)
            case "/config":
                print_config(args.provider, args.model or "(default)", args.opa_url)
            case _:
                try:
                    response = agent.run(user_input)
                    print(f"\n  🤖 {response}\n")
                except Exception as e:
                    print(f"\n  ❌ Error: {e}\n")


if __name__ == "__main__":
    main()
