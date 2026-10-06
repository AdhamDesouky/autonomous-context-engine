from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.retrieval.vector_store import HybridVectorStore


def format_search_results(results: list[Any]) -> str:
    if not results:
        return "No relevant documentation found."

    formatted_results = []
    for index, result in enumerate(results, 1):
        payload = result.payload or {}
        source = payload.get("source_file", "Unknown source")
        page = payload.get("page", "Unknown")
        heading = payload.get("heading", "General")
        text = payload.get("text", "")
        formatted_results.append(
            f"--- Result {index} ---\n"
            f"Source: {source} (Page {page})\n"
            f"Section: {heading}\n"
            f"Content: {text}\n"
        )
    return "\n".join(formatted_results)


def create_mcp_server(store: HybridVectorStore | None = None) -> FastMCP:
    mcp = FastMCP("Enterprise_RAG_ReadOnly")

    def get_store() -> HybridVectorStore:
        nonlocal store
        if store is None:
            store = HybridVectorStore()
        return store

    @mcp.tool()
    def query_documentation(query: str, num_results: int = 3) -> str:
        """Search indexed documentation and return source metadata and snippets."""
        normalized_query = query.strip()
        if len(normalized_query) < 2:
            raise ValueError("query must contain at least 2 non-whitespace characters")
        if not 1 <= num_results <= 10:
            raise ValueError("num_results must be between 1 and 10")

        results = get_store().hybrid_search(normalized_query, limit=num_results)
        return format_search_results(results)

    @mcp.tool()
    def list_indexed_documents() -> str:
        """List source filenames currently represented in the knowledge base."""
        documents = sorted(get_store().get_indexed_files())
        return "\n".join(documents) if documents else "No indexed documents found."

    return mcp


if __name__ == "__main__":
    create_mcp_server().run()
