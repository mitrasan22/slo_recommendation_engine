"""
Shared MCP client configuration and helpers.

Notes
-----
Mirrors the pattern used in the ``agentic_rgm`` package by centralising the
knowledge MCP server connection configuration in one module. It also provides
an async helper for direct MCP tool calls when the caller needs deterministic
tool execution and immediate access to the parsed result.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from typing import Any, cast

from dotenv import load_dotenv
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

logger = logger.bind(name=__name__)

MCP_SERVER_PYTHON = os.getenv("MCP_SERVER_PYTHON", sys.executable)
MCP_SESSION_TIMEOUT = float(os.getenv("MCP_SESSION_TIMEOUT", "60"))

knowledge_server_params = StdioServerParameters(
    command=MCP_SERVER_PYTHON,
    args=["-m", "slo_engine.mcp.knowledge_mcp_server"],
    env={**os.environ},
)

knowledge_server = StdioConnectionParams(
    server_params=knowledge_server_params,
    timeout=MCP_SESSION_TIMEOUT,
)


class KnowledgeMCPClient:
    """Persistent stdio MCP client for the knowledge server."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def _connect(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
        stack = contextlib.AsyncExitStack()
        read, write = await stack.enter_async_context(
            stdio_client(knowledge_server_params)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._exit_stack = stack
        self._session = session

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if self._session is None:
                await self._connect()

        for attempt in range(2):
            try:
                assert self._session is not None
                result = await self._session.call_tool(tool_name, arguments=arguments)
                if getattr(result, "structuredContent", None):
                    return cast(dict[str, Any], result.structuredContent)
                parts = []
                for item in (result.content or []):
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(text)
                text_payload = "\n".join(parts).strip()
                if not text_payload:
                    logger.warning(
                        "Knowledge MCP tool '{}' returned empty text content. isError={} content={}",
                        tool_name,
                        getattr(result, "isError", False),
                        result.content,
                    )
                    return {}
                return json.loads(text_payload)
            except Exception as exc:
                if attempt == 1:
                    logger.error(
                        "Knowledge MCP call failed after reconnect: {} | tool={} | raw_payload={!r}",
                        exc,
                        tool_name,
                        locals().get("text_payload", ""),
                    )
                    return {}
                logger.warning(
                    "Knowledge MCP call failed ({}), reconnecting... | tool={} | raw_payload={!r}",
                    exc,
                    tool_name,
                    locals().get("text_payload", ""),
                )
                async with self._lock:
                    self._session = None
                    await self._connect()
        return {}

    async def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 4,
        doc_type: str = "all",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "retrieve_knowledge",
            {"query": query, "top_k": top_k, "doc_type": doc_type},
        )


knowledge_client = KnowledgeMCPClient()
