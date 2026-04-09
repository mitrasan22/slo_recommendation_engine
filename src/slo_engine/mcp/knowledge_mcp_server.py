"""
Knowledge RAG MCP Server exposing SRE runbooks, templates, and incidents.

Notes
-----
Single source of truth: all documents live in ``data/knowledge_base/*.json``
and ``docs/*.md``. This server delegates entirely to ``KnowledgeStore`` — it
holds no inline knowledge base of its own.

Tools:
  ``retrieve_knowledge``  — semantic MMR retrieval using sentence-transformers.
  ``get_slo_template``    — template lookup by service type tag.
  ``list_document_types`` — introspect available document type counts.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.types import CallToolResult, ListToolsRequest, ListToolsResult, TextContent, Tool

from slo_engine.rag.knowledge_store import knowledge_store

logger = logging.getLogger(__name__)
app = Server("knowledge-mcp-server")


@app.list_tools()
async def list_tools(_: ListToolsRequest) -> ListToolsResult:
    """
    Return the list of tools exposed by the knowledge MCP server.

    Parameters
    ----------
    _ : ListToolsRequest
        MCP list-tools request (unused).

    Returns
    -------
    ListToolsResult
        MCP result containing the three knowledge tool definitions.

    Notes
    -----
    Tool schemas include the live document count from ``knowledge_store``
    in the ``retrieve_knowledge`` description for observability.
    """
    return ListToolsResult(tools=[
        Tool(
            name="retrieve_knowledge",
            description=(
                "Retrieve relevant SRE runbooks, SLO templates, incidents, and guidelines "
                "for a given query using semantic search with MMR diversity. "
                f"Knowledge base: {knowledge_store.document_count} documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of results to return",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["runbook", "template", "incident", "guideline",
                                 "documentation", "all"],
                        "default": "all",
                        "description": "Filter by document type",
                    },
                    "mmr_lambda": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "MMR relevance vs diversity trade-off (0-1). "
                            "Omit for automatic intent-based selection. "
                            "1.0 = pure relevance, 0.0 = pure diversity."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_slo_template",
            description="Get SLO template(s) for a specific service type by tag match.",
            inputSchema={
                "type": "object",
                "properties": {
                    "service_type": {
                        "type": "string",
                        "description": (
                            "Service type keyword, e.g. 'api_gateway', 'database', "
                            "'payment', 'auth', 'kafka', 'redis', 'ml_inference'"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum number of templates to return",
                    },
                },
                "required": ["service_type"],
            },
        ),
        Tool(
            name="list_document_types",
            description="Introspect the knowledge base — returns document count per type.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ])


@app.call_tool()
async def call_tool(tool_name: str, args: dict | None) -> CallToolResult:
    """
    Dispatch an MCP tool call to the corresponding knowledge tool implementation.

    Parameters
    ----------
    tool_name : str
        MCP tool name.
    args : dict or None
        Tool arguments passed by the MCP client.

    Returns
    -------
    CallToolResult
        MCP result with a single ``TextContent`` item containing the JSON
        response, or an error JSON on failure.

    Notes
    -----
    ``retrieve_knowledge`` uses ``knowledge_store.retrieve`` with optional
    ``doc_type`` filter and ``mmr_lambda`` override. ``get_slo_template`` is
    implemented as a ``retrieve_knowledge`` call restricted to ``doc_type="template"``
    with ``mmr_lambda=0.80`` for high-relevance lookup. ``list_document_types``
    accesses ``knowledge_store._docs`` directly for introspection.
    """
    args = args or {}

    try:
        if tool_name == "retrieve_knowledge":
            query      = str(args.get("query", ""))
            top_k      = int(args.get("top_k", 3))
            doc_type   = str(args.get("doc_type", "all"))
            mmr_lambda = args.get("mmr_lambda")

            if mmr_lambda is not None:
                mmr_lambda = float(mmr_lambda)

            results = knowledge_store.retrieve(
                query=query,
                top_k=top_k,
                doc_type=doc_type,
                mmr_lambda=mmr_lambda,
            )
            source_ids = [r["id"] for r in results]
            context_lines = [
                f"[{r['id']} — {r['type'].upper()}] {r['title']}\n{r['content']}"
                for r in results
            ]
            context_summary = "\n\n".join(context_lines)
            payload = {
                "results":         results,
                "source_ids":      source_ids,
                "context_summary": context_summary,
                "query":           query,
                "total_returned":  len(results),
                "kb_size":         knowledge_store.document_count,
            }
            result = json.dumps(payload)

        elif tool_name == "get_slo_template":
            service_type = str(args.get("service_type", "")).lower()
            top_k        = int(args.get("top_k", 3))

            query   = f"{service_type} SLO template availability latency"
            results = knowledge_store.retrieve(
                query=query,
                top_k=top_k,
                doc_type="template",
                mmr_lambda=0.80,
            )
            payload = {
                "templates":    results,
                "service_type": service_type,
                "found":        len(results) > 0,
            }
            result = json.dumps(payload)

        elif tool_name == "list_document_types":
            all_docs = knowledge_store._docs   # noqa: SLF001
            type_counts = dict(Counter(d["type"] for d in all_docs))
            payload = {
                "total":       knowledge_store.document_count,
                "by_type":     type_counts,
                "sources":     ["data/knowledge_base/*.json", "docs/*.md"],
            }
            result = json.dumps(payload)

        else:
            payload = {"error": f"Unknown tool: {tool_name}"}
            result = json.dumps(payload)

        return CallToolResult(
            content=[TextContent(type="text", text=result)],
            structuredContent=payload,
        )

    except Exception as e:
        logger.error("Knowledge MCP error: tool=%s error=%s", tool_name, e)
        error_payload = {"error": str(e)}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(error_payload))],
            structuredContent=error_payload,
            isError=True,
        )


async def main() -> None:
    """
    Start the knowledge MCP server using stdio transport.

    Returns
    -------
    None

    Notes
    -----
    Runs the MCP server loop until the stdio streams are closed. Server name
    is ``"knowledge-mcp-server"`` at version ``"1.0.0"``.
    """
    async with stdio_server() as (r, w):
        await app.run(
            r, w,
            InitializationOptions(
                server_name="knowledge-mcp-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
