"""Policy engine — abstract interface + OPA client + allow-all fallback."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy decision
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""
    allow: bool
    reason: str = ""
    timeout_seconds: float = 30.0   # max execution time for this tool call
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class PolicyClient(ABC):
    """Interface for policy engines — swap OPA for anything else."""

    @abstractmethod
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        """Evaluate whether a tool call is permitted."""
        ...


# ---------------------------------------------------------------------------
# Policy context — attributes for ABAC
# ---------------------------------------------------------------------------

@dataclass
class PolicyContext:
    """Contextual attributes sent to OPA for ABAC evaluation."""
    skill: str = "core"
    user_role: str = "user"
    data_sensitivity: str = "low"       # low, medium, high
    session_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_input(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Build the OPA input document."""
        return {
            "tool": tool_name,
            "args": arguments,
            "skill": self.skill,
            "user_role": self.user_role,
            "data_sensitivity": self.data_sensitivity,
            "session_id": self.session_id,
            "timestamp": time.time(),
            **self.extra,
        }


# ---------------------------------------------------------------------------
# OPA client
# ---------------------------------------------------------------------------

class OPAClient(PolicyClient):
    """Evaluates policy against a running OPA server via REST API."""

    def __init__(
        self,
        opa_url: str = "http://localhost:8181",
        policy_path: str = "v1/data/secureclaw/authz",
        default_timeout: float = 30.0,
    ):
        self.opa_url = opa_url.rstrip("/")
        self.policy_path = policy_path
        self.default_timeout = default_timeout

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        import httpx

        ctx = context or PolicyContext()
        opa_input = ctx.to_input(tool_name, arguments)

        url = f"{self.opa_url}/{self.policy_path}"

        try:
            resp = httpx.post(
                url,
                json={"input": opa_input},
                timeout=5.0,  # OPA itself should reply fast
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})

            allow = result.get("allow", False)
            reason = result.get("reason", "")
            timeout = result.get("timeout_seconds", self.default_timeout)

            decision = PolicyDecision(
                allow=allow,
                reason=reason,
                timeout_seconds=timeout,
                metadata=result,
            )

        except httpx.ConnectError:
            logger.error(f"Cannot reach OPA at {self.opa_url} — denying by default")
            decision = PolicyDecision(
                allow=False,
                reason="OPA unreachable — fail-closed",
            )

        except Exception as e:
            logger.error(f"OPA evaluation error: {e} — denying by default")
            decision = PolicyDecision(
                allow=False,
                reason=f"Policy evaluation failed: {e}",
            )

        logger.info(
            f"Policy: tool={tool_name} allow={decision.allow} reason={decision.reason}"
        )
        return decision


# ---------------------------------------------------------------------------
# Allow-all fallback (dev/testing)
# ---------------------------------------------------------------------------

class AllowAllPolicy(PolicyClient):
    """Permits everything — for local dev only."""

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(allow=True, reason="allow-all policy (dev mode)")
