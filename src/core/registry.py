"""Tool registry — abstract interface + SQLite backend."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """A registered tool definition."""
    name: str
    description: str
    input_schema: dict[str, Any]      # JSON Schema for parameters
    handler: Callable[..., Any]        # the actual function to call
    skill: str = "core"                # which skill owns this tool
    version: str = "0.1.0"
    tags: list[str] = field(default_factory=list)

    def to_llm_schema(self) -> dict[str, Any]:
        """Return Anthropic-style tool definition for the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ToolRegistry(ABC):
    """Interface for tool registries — swap backends without touching the loop."""

    @abstractmethod
    def register(self, tool: ToolDef) -> None:
        """Register or update a tool."""
        ...

    @abstractmethod
    def get(self, name: str) -> ToolDef | None:
        """Look up a tool by name."""
        ...

    @abstractmethod
    def list_tools(self, skill: str | None = None) -> list[ToolDef]:
        """List all tools, optionally filtered by skill."""
        ...

    @abstractmethod
    def remove(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        ...

    def get_llm_schemas(self, skill: str | None = None) -> list[dict[str, Any]]:
        """Return all tool schemas in the format the LLM expects."""
        return [t.to_llm_schema() for t in self.list_tools(skill)]


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class SQLiteToolRegistry(ToolRegistry):
    """Persistent tool registry backed by SQLite.

    Tool metadata (name, description, schema, skill, version, tags) is
    persisted. Handlers (callables) are held in memory and must be
    re-registered on restart — this is by design since you can't serialise
    arbitrary functions safely.
    """

    def __init__(self, db_path: str | Path = "tools.db"):
        self._db_path = str(db_path)
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    name         TEXT PRIMARY KEY,
                    description  TEXT NOT NULL,
                    input_schema TEXT NOT NULL,
                    skill        TEXT NOT NULL DEFAULT 'core',
                    version      TEXT NOT NULL DEFAULT '0.1.0',
                    tags         TEXT NOT NULL DEFAULT '[]'
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def register(self, tool: ToolDef) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tools (name, description, input_schema, skill, version, tags)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description  = excluded.description,
                    input_schema = excluded.input_schema,
                    skill        = excluded.skill,
                    version      = excluded.version,
                    tags         = excluded.tags
                """,
                (
                    tool.name,
                    tool.description,
                    json.dumps(tool.input_schema),
                    tool.skill,
                    tool.version,
                    json.dumps(tool.tags),
                ),
            )
        self._handlers[tool.name] = tool.handler

    def get(self, name: str) -> ToolDef | None:
        handler = self._handlers.get(name)
        if handler is None:
            return None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT name, description, input_schema, skill, version, tags FROM tools WHERE name = ?",
                (name,),
            ).fetchone()

        if row is None:
            return None

        return ToolDef(
            name=row[0],
            description=row[1],
            input_schema=json.loads(row[2]),
            handler=handler,
            skill=row[3],
            version=row[4],
            tags=json.loads(row[5]),
        )

    def list_tools(self, skill: str | None = None) -> list[ToolDef]:
        with self._connect() as conn:
            if skill:
                rows = conn.execute(
                    "SELECT name, description, input_schema, skill, version, tags FROM tools WHERE skill = ?",
                    (skill,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT name, description, input_schema, skill, version, tags FROM tools"
                ).fetchall()

        results: list[ToolDef] = []
        for row in rows:
            handler = self._handlers.get(row[0])
            if handler is None:
                continue  # metadata exists but handler not loaded — skip
            results.append(ToolDef(
                name=row[0],
                description=row[1],
                input_schema=json.loads(row[2]),
                handler=handler,
                skill=row[3],
                version=row[4],
                tags=json.loads(row[5]),
            ))
        return results

    def remove(self, name: str) -> bool:
        self._handlers.pop(name, None)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM tools WHERE name = ?", (name,))
            return cursor.rowcount > 0
