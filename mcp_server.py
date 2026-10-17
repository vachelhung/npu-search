"""MCP server exposing the NPU semantic index to Claude Code (stdio transport)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from npusearch import index  # noqa: E402

mcp = MCPServer("npu-search")
_searcher = None


def searcher() -> index.Searcher:
    # Compiling for the NPU takes ~15s, so do it on first use rather than at startup.
    global _searcher
    if _searcher is None:
        _searcher = index.Searcher()
    return _searcher


@mcp.tool()
def semantic_search(query: str, k: int = 5, path_contains: str | None = None) -> list[dict]:
    """Search the user's local documents and code by meaning (Chinese or English).

    Covers the folders listed under "roots" in config.json: PDFs, Word/Excel/PowerPoint,
    Markdown, text, notebooks, source code and KiCad files.
    Returns the best-matching passage per file with its path and a cosine score
    (roughly: >0.85 strong, 0.80-0.85 related, <0.80 weak). Read the file itself
    for full context. Embeddings run on the laptop's Intel NPU.

    Args:
        query: what to look for, phrased naturally.
        k: number of files to return.
        path_contains: optional case-insensitive substring to restrict paths (e.g. "DataSheet").
    """
    return searcher().search(query, k, path_contains)


@mcp.tool()
def index_status() -> dict:
    """Number of indexed files and chunks, files that failed to parse, and last index time."""
    return index.status()


@mcp.tool()
def reindex() -> dict:
    """Incrementally re-index: new or changed files are embedded, deleted files removed."""
    return index.build(verbose=False)


if __name__ == "__main__":
    mcp.run()
