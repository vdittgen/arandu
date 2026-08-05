"""Action executor — executes confirmed MCP actions via the MCP client.

Takes a fully-specified action proposal (connector command, tool name,
arguments) and executes it via a fresh MCP client connection.

sensitivity_tier: 2 (executes side-effect actions on behalf of the user)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.extensions.mcp.client import (
    McpClient,
    McpConnectionError,
    McpTimeoutError,
    McpToolError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionResult:
    """Result of executing an MCP action.

    sensitivity_tier: 2
    """

    proposal_id: str
    status: str  # "success" | "error"
    output: str  # Human-readable result summary
    raw_result: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# ActionExecutor
# ---------------------------------------------------------------------------


class ActionExecutor:
    """Executes confirmed MCP actions via the MCP client.

    Creates a fresh MCP client connection for each action execution.
    All execution info is passed explicitly — no retained state.

    sensitivity_tier: 2
    """

    def __init__(
        self,
        mcp_client_factory: Callable[
            [str, tuple[str, ...], float], Any
        ] | None = None,
        mcp_timeout: float = 30.0,
    ) -> None:
        """Initialise the action executor.

        Args:
            mcp_client_factory: Creates an MCP client given
                ``(command, args, timeout)``. Must support the
                context-manager protocol. Defaults to ``McpClient``.
            mcp_timeout: Timeout in seconds for MCP operations.

        sensitivity_tier: 1
        """
        self._factory = mcp_client_factory or self._default_factory
        self._timeout = mcp_timeout

    def execute(
        self,
        connector_id: str,
        command: str,
        args: tuple[str, ...],
        tool_name: str,
        arguments: dict[str, Any],
        proposal_id: str,
    ) -> ActionResult:
        """Execute a confirmed MCP action.

        Opens a fresh MCP client, calls the tool, and returns the result.

        Args:
            connector_id: The connector that owns the tool.
            command: MCP server command (e.g. "npx").
            args: MCP server arguments (e.g. ("-y", "@supermemoryai/...")).
            tool_name: The MCP tool to call.
            arguments: Tool parameters.
            proposal_id: UUID from the original proposal for tracking.

        Returns:
            ActionResult with status and output.

        sensitivity_tier: 2
        """
        # WhatsApp send_message can't go through MCP because the
        # catalog command (``node client.js``) is the listener — not
        # an MCP server. Per CLAUDE.md only one Baileys connection per
        # phone exists, so the running listener owns the socket and
        # all sends route through its outbox IPC.
        if connector_id == "whatsapp" and tool_name == "send_message":
            return self._execute_whatsapp_send(arguments, proposal_id)

        try:
            with self._factory(command, args, self._timeout) as client:
                raw_result = client.call_tool(tool_name, arguments)
        except McpToolError as exc:
            logger.warning(
                "MCP tool %s failed for %s: %s",
                tool_name, connector_id, exc,
            )
            return ActionResult(
                proposal_id=proposal_id,
                status="error",
                output=f"Tool call failed: {exc}",
                error=str(exc),
            )
        except McpConnectionError as exc:
            logger.warning(
                "MCP connection failed for %s: %s",
                connector_id, exc,
            )
            return ActionResult(
                proposal_id=proposal_id,
                status="error",
                output=f"Could not connect to {connector_id}: {exc}",
                error=str(exc),
            )
        except McpTimeoutError as exc:
            logger.warning(
                "MCP timeout for %s tool %s: %s",
                connector_id, tool_name, exc,
            )
            return ActionResult(
                proposal_id=proposal_id,
                status="error",
                output=f"Action timed out: {exc}",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error executing %s on %s",
                tool_name, connector_id,
            )
            return ActionResult(
                proposal_id=proposal_id,
                status="error",
                output=f"Unexpected error: {exc}",
                error=str(exc),
            )

        output = _summarize_result(tool_name, raw_result)
        return ActionResult(
            proposal_id=proposal_id,
            status="success",
            output=output,
            raw_result=raw_result,
        )

    def _execute_whatsapp_send(
        self,
        arguments: dict[str, Any],
        proposal_id: str,
    ) -> ActionResult:
        """Route a WhatsApp send through the listener's outbox IPC.

        The listener owns the sole Baileys connection. The catalog's
        ``node client.js`` command is the listener itself and doesn't
        implement the MCP protocol, so a tools/call request bounces
        back as ``Unknown command`` and the action fails. Routing
        through ``send_text_via_running_listener`` writes a request to
        the outbox directory and the listener's polling loop picks it
        up.

        sensitivity_tier: 3
        """
        to = str(arguments.get("to") or "").strip()
        text = str(
            arguments.get("text") or arguments.get("body") or "",
        ).strip()
        if not to or not text:
            return ActionResult(
                proposal_id=proposal_id,
                status="error",
                output="WhatsApp send missing 'to' or 'text'",
                error="missing required field",
            )

        try:
            from src.extensions.bridges.whatsapp.listener import (
                send_text_via_running_listener,
            )
        except Exception as exc:  # noqa: BLE001
            return ActionResult(
                proposal_id=proposal_id, status="error",
                output=f"WhatsApp listener module unavailable: {exc}",
                error=str(exc),
            )

        # Baileys JIDs are pure digits; the listener wrapper accepts a
        # plain phone number too but stripping ``+`` keeps the JID
        # canonical and matches what the node-side ``cmd: send`` path
        # does on the listener.
        to_jid = to if "@" in to else f"{to.lstrip('+')}@s.whatsapp.net"

        response = send_text_via_running_listener(
            to=to_jid, message=text, timeout_seconds=self._timeout,
        )
        if response is None:
            return ActionResult(
                proposal_id=proposal_id, status="error",
                output=(
                    "WhatsApp listener is not running — start it from "
                    "Settings → Connectors"
                ),
                error="listener_not_running",
            )

        status = str(response.get("status") or "").strip().lower()
        if status == "sent":
            resolved = response.get("resolved_jid") or to_jid
            display_to = to
            if resolved and resolved != to_jid:
                # Strip the JID suffix when reporting so the user sees
                # a phone number rather than a Baileys-internal string.
                display_to = resolved.split("@", 1)[0]
            return ActionResult(
                proposal_id=proposal_id, status="success",
                output=f"Sent WhatsApp message to {display_to}",
                raw_result=[{
                    "to": resolved,
                    "message_id": response.get("message_id"),
                }],
            )

        err = str(response.get("error") or "Send failed")
        return ActionResult(
            proposal_id=proposal_id, status="error",
            output=f"WhatsApp send failed: {err}", error=err,
        )

    @staticmethod
    def _default_factory(
        command: str,
        args: tuple[str, ...],
        timeout: float,
    ) -> McpClient:
        """Default MCP client factory.

        sensitivity_tier: 1
        """
        return McpClient(command=command, args=args, timeout=timeout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize_result(
    tool_name: str,
    raw_result: list[dict[str, Any]],
) -> str:
    """Build a human-readable summary of an MCP tool result.

    sensitivity_tier: 2
    """
    display = tool_name.replace("_", " ").title()

    if not raw_result:
        return f"{display} completed successfully."

    # Try to extract a useful summary from the first result
    first = raw_result[0]
    if "_raw_text" in first:
        text = str(first["_raw_text"])[:200]
        return f"{display}: {text}"

    # Look for common response fields
    for key in ("message", "status", "result", "id", "title", "name"):
        if key in first:
            return f"{display}: {first[key]}"

    return f"{display} completed successfully ({len(raw_result)} result(s))."
