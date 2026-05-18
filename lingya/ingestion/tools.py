from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lingya.llm.base import ToolDefinition, ToolParameter

if TYPE_CHECKING:
    from lingya.memory.manager import MemoryManager

FETCH_URL_TOOL = ToolDefinition(
    name="fetch_url",
    description="Fetch and ingest the content of a web page by URL",
    parameters=[
        ToolParameter(name="url", type="string", description="The URL to fetch", required=True),
    ],
)

INGEST_TOOL = ToolDefinition(
    name="ingest_content",
    description="Store a piece of text into long-term memory for future recall",
    parameters=[
        ToolParameter(name="content", type="string", description="The text content to store", required=True),
        ToolParameter(name="source", type="string", description="Source description (e.g. URL, filename)", required=True),
    ],
)

ALL_TOOLS = [FETCH_URL_TOOL, INGEST_TOOL]


async def execute_tool(
    name: str,
    args: dict[str, Any],
    memory_manager: MemoryManager,
) -> str:
    match name:
        case "fetch_url":
            url = args.get("url", "")
            if not url:
                return "Error: URL is required"
            from .loader import ingest_url
            chunk_ids = await ingest_url(memory_manager, url)
            return f"Fetched and ingested {len(chunk_ids)} chunks from {url}"

        case "ingest_content":
            content = args.get("content", "")
            source = args.get("source", "manual")
            if not content:
                return "Error: content is required"
            chunk_ids = await memory_manager.ingest_content(content, source, "manual")
            return f"Ingested {len(chunk_ids)} chunks"

        case _:
            return f"Unknown tool: {name}"
