from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from bs4 import BeautifulSoup

from .chunker import chunk_text

if TYPE_CHECKING:
    from lingya.memory.manager import MemoryManager


async def ingest_text(
    memory_manager: MemoryManager,
    text: str,
    source: str,
    content_type: str,
) -> list[str]:
    return await memory_manager.ingest_content(text, source, content_type)


async def ingest_file(
    memory_manager: MemoryManager,
    file_path: str,
) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    content_type = _detect_type(path.suffix)
    return await memory_manager.ingest_content(text, str(path), content_type)


async def ingest_url(
    memory_manager: MemoryManager,
    url: str,
) -> list[str]:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "LingYa/0.1"})
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)

    return await memory_manager.ingest_content(text, url, "web_page")


def _detect_type(suffix: str) -> str:
    mapping = {
        ".txt": "text_file",
        ".md": "markdown",
        ".py": "code",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".pdf": "pdf",
        ".html": "html",
        ".csv": "csv",
    }
    return mapping.get(suffix.lower(), "file")
