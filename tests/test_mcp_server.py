import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.mcp.server import create_mcp_server, format_search_results


def empty_search(query, limit):
    _ = (query, limit)
    return []


def test_format_search_results_includes_citation_metadata():
    results = [
        SimpleNamespace(
            payload={
                "source_file": "manual.pdf",
                "page": 4,
                "heading": "Security",
                "text": "Use authenticated communication.",
            }
        )
    ]

    formatted = format_search_results(results)

    assert "manual.pdf (Page 4)" in formatted
    assert "Section: Security" in formatted
    assert "Use authenticated communication." in formatted


def test_format_search_results_handles_empty_results():
    assert format_search_results([]) == "No relevant documentation found."


def test_mcp_tools_reject_invalid_search_limits():
    server = create_mcp_server(
        SimpleNamespace(
            hybrid_search=empty_search,
            get_indexed_files=lambda: set(),
        )
    )
    tools = server._tool_manager.list_tools()
    query_tool = next(tool for tool in tools if tool.name == "query_documentation")

    with pytest.raises(ValueError, match="between 1 and 10"):
        query_tool.fn("valid query", 11)


def test_mcp_server_exposes_only_read_only_tools():
    server = create_mcp_server(
        SimpleNamespace(
            hybrid_search=empty_search,
            get_indexed_files=lambda: set(),
        )
    )

    tool_names = {tool.name for tool in server._tool_manager.list_tools()}

    assert tool_names == {"query_documentation", "list_indexed_documents"}
